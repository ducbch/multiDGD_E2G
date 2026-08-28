"""In-silico perturbation engine: silence each candidate feature and record the mean
per-response-feature change (delta).

Mini-batched over cells + sparse-sliced so peak GPU memory is O(cell_batch), not O(n_cells): the full
count matrix stays on CPU (only per-chunk silenced-modality slices go to the GPU), and every
decode runs in cell chunks. This is exact vs the whole-matrix version (verified per-pair):
the decoder is per-cell (no cross-cell op), the delta is a linear cell-mean (chunk sums
combined), and the rep refit accumulates gradients across chunks before ONE optimizer step
(== the full-batch gradient). Only float32 summation order differs (~1e-6, negligible).

The result is null-agnostic and cacheable: change the scoring null/sidedness without recomputing.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def compute_deltas(adapter, groups, silenced_index, resp_index, direction,
                   steps=1, min_cells=3, cell_batch=8192, log=print):
    """Silence each candidate feature in turn and return its mean per-response-feature deltas.

    Parameters
    ----------
    adapter        : MultiDGDAdapter wrapping a trained model with test reps learned.
    groups         : {silenced_feature -> [response_feature, ...]} candidate targets.
    silenced_index : silenced-feature name -> within-modality index of the SILENCED feature.
    resp_index     : response-feature name -> within-modality index in the RESPONSE modality.
    direction      : "gene2peak" (silence gene mod 0, read peaks) or
                     "peak2gene" (silence peak mod 1, read genes).
    steps          : rep-refit gradient steps per silencing (multiDGD default 1).
    min_cells      : skip features detected in fewer than this many cells.
    cell_batch     : cells per decode chunk (bounds GPU memory; exact regardless of value).

    Returns (D, kept, p0, resp_names):
      D          float32 (n_kept x n_resp) mean delta per response feature
      kept       silenced-feature names in D row order (those passing min_cells)
      p0         float32 (n_resp,) baseline predicted softmax share per response feature
      resp_names response-feature names in D column order (for cache validation)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    silence_mod = 0 if direction == "gene2peak" else 1
    resp_mod = 1 - silence_mod
    switch = adapter.modality_switch

    x_cpu = adapter.counts                        # n x total, CPU dense (NOT moved to GPU)
    n = x_cpu.shape[0]
    sil_lo, sil_hi = (0, switch) if silence_mod == 0 else (switch, x_cpu.shape[1])

    lib = adapter.library.to(device)              # n x 2 (tiny)
    rep = adapter.test_rep
    has_corr = adapter.has_correction
    corr_z = adapter.correction_z                 # detached; not optimized
    z0 = rep.z.detach().clone()                   # baseline latent (small)

    def chunks():
        """Yield (start, stop) cell-row ranges of at most ``cell_batch`` cells each."""
        for lo in range(0, n, cell_batch):
            yield lo, min(lo + cell_batch, n)

    # --- baseline response-feature share p0 (chunked; no full baseline decode held) ---
    with torch.no_grad():
        n_resp = adapter.decode_z(z0[:1], corr_z[:1] if has_corr else None)[resp_mod].shape[1]
        p0_sum = torch.zeros(n_resp, device=device)
        for lo, hi in chunks():
            yb = adapter.decode_z(z0[lo:hi], corr_z[lo:hi] if has_corr else None)
            p0_sum += yb[resp_mod].sum(0)
        p0 = (p0_sum / n).cpu().numpy().astype(np.float32)

    resp_names = [None] * n_resp
    for name, j in resp_index.items():
        resp_names[j] = name

    silenced_features = list(groups.keys())
    D = np.empty((len(silenced_features), n_resp), dtype=np.float32)
    kept = []
    row = 0
    t0 = time.time()
    for ai, silenced in enumerate(silenced_features):
        f = silenced_index[silenced]
        comb = f if silence_mod == 0 else switch + f
        mask_cpu = x_cpu[:, comb] > 0             # cells where the silenced feature is detected
        if int(mask_cpu.sum()) < min_cells:
            continue

        # --- rep refit: chunked grad accumulation, then ONE optimizer step ---
        opt = torch.optim.Adam(rep.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.5, 0.7))
        for _ in range(steps):
            opt.zero_grad()
            for lo, hi in chunks():
                tgt = x_cpu[lo:hi, sil_lo:sil_hi].to(device).clone()  # only this chunk+modality
                tgt[:, f] = 0.0                                       # silence the feature
                yb = adapter.decode_z(rep.z[lo:hi], corr_z[lo:hi] if has_corr else None)
                loss = adapter.decoder_loss(yb[silence_mod], tgt,
                                            scale=lib[lo:hi, silence_mod].unsqueeze(1),
                                            mod_id=silence_mod, gene_id=f)
                loss.backward()
                del yb, tgt
            opt.step()

        # --- delta = mean over detected cells of (perturbed - baseline), chunked ---
        with torch.no_grad():
            sum_d = torch.zeros(n_resp, device=device)
            cnt = 0
            for lo, hi in chunks():
                m = mask_cpu[lo:hi]
                if not bool(m.any()):
                    continue
                m_dev = m.to(device)
                y_pert = adapter.decode_z(rep.z[lo:hi], corr_z[lo:hi] if has_corr else None)[resp_mod]
                y_base = adapter.decode_z(z0[lo:hi], corr_z[lo:hi] if has_corr else None)[resp_mod]
                sum_d += (y_pert - y_base)[m_dev].sum(0)
                cnt += int(m.sum())
                del y_pert, y_base, m_dev
            d = (sum_d / cnt).cpu().numpy()

        rep.z.data.copy_(z0)                      # restore baseline representation
        D[row] = d; kept.append(silenced); row += 1
        if (ai + 1) % 500 == 0:
            log(f"  {ai + 1}/{len(silenced_features)} silenced features ({(time.time() - t0) / 60:.1f} min)")
    return D[:row], kept, p0, resp_names


def compute_cell_deltas(adapter, silenced, silenced_index, resp_cols, direction,
                        steps=1, cell_batch=8192):
    """Per-cell deltas for a SINGLE silenced feature (for a per-cell x per-target heatmap).

    Same silencing + rep-refit as `compute_deltas`, but keeps the per-cell rows instead of
    averaging them, and only for the given response columns. Cheap for one silenced feature.

    Parameters mirror `compute_deltas`; `resp_cols` is the list/array of response-modality
    indices to keep (e.g. the candidate peaks of one gene).

    Returns (delta_cells, cell_idx):
      delta_cells float32 (n_detected x len(resp_cols)) perturbed - baseline per detected cell
      cell_idx    int array of the detected cell row indices (into adapter.counts rows)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    silence_mod = 0 if direction == "gene2peak" else 1
    resp_mod = 1 - silence_mod
    switch = adapter.modality_switch
    resp_cols = np.asarray(list(resp_cols), dtype=np.int64)

    x_cpu = adapter.counts
    n = x_cpu.shape[0]
    sil_lo, sil_hi = (0, switch) if silence_mod == 0 else (switch, x_cpu.shape[1])
    lib = adapter.library.to(device)
    rep = adapter.test_rep
    has_corr = adapter.has_correction
    corr_z = adapter.correction_z
    z0 = rep.z.detach().clone()

    def chunks():
        """Yield (start, stop) cell-row ranges of at most ``cell_batch`` cells each."""
        for lo in range(0, n, cell_batch):
            yield lo, min(lo + cell_batch, n)

    f = silenced_index[silenced]
    comb = f if silence_mod == 0 else switch + f
    mask_cpu = x_cpu[:, comb] > 0
    cell_idx = np.nonzero(mask_cpu.cpu().numpy() if hasattr(mask_cpu, "cpu") else np.asarray(mask_cpu))[0]

    opt = torch.optim.Adam(rep.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.5, 0.7))
    for _ in range(steps):
        opt.zero_grad()
        for lo, hi in chunks():
            tgt = x_cpu[lo:hi, sil_lo:sil_hi].to(device).clone()
            tgt[:, f] = 0.0
            yb = adapter.decode_z(rep.z[lo:hi], corr_z[lo:hi] if has_corr else None)
            loss = adapter.decoder_loss(yb[silence_mod], tgt,
                                        scale=lib[lo:hi, silence_mod].unsqueeze(1),
                                        mod_id=silence_mod, gene_id=f)
            loss.backward()
            del yb, tgt
        opt.step()

    cols = torch.as_tensor(resp_cols, device=device)
    rows = []
    with torch.no_grad():
        for lo, hi in chunks():
            m = mask_cpu[lo:hi]
            if not bool(m.any() if hasattr(m, "any") else np.any(m)):
                continue
            m_dev = (m.to(device) if hasattr(m, "to") else torch.as_tensor(np.asarray(m), device=device))
            y_pert = adapter.decode_z(rep.z[lo:hi], corr_z[lo:hi] if has_corr else None)[resp_mod]
            y_base = adapter.decode_z(z0[lo:hi], corr_z[lo:hi] if has_corr else None)[resp_mod]
            diff = (y_pert - y_base)[:, cols][m_dev]
            rows.append(diff.cpu().numpy())
            del y_pert, y_base, m_dev

    rep.z.data.copy_(z0)
    delta_cells = np.concatenate(rows, axis=0).astype(np.float32) if rows \
        else np.empty((0, len(resp_cols)), dtype=np.float32)
    return delta_cells, cell_idx
