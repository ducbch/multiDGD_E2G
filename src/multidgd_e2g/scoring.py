"""Score perturbation deltas into calibrated enhancer-gene link statistics.

The p-value uses an empirical background null (the same idea Signac LinkPeaks and FigR
use), so the raw delta need not follow any distribution: for each silenced feature, the cis targets'
delta is z-scored against a background set of distal response features. `pvalue` is 1-sided
(activating tail: silencing an activator lowers the partner's predicted signal, delta < 0, so
z = (mean(delta_bg) - delta_cis)/sd(delta_bg) is large positive); `twosided_pvalue` is also
emitted. This step is pure CPU and null-agnostic vs the cached deltas -- changing the null
(random vs accessibility-matched), n_bg, bins, or sidedness only reruns this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-12


def score_from_deltas(D, kept, p0, groups, resp_index, direction,
                      n_bg=500, seed=0, match_bg=True, n_bins=25, log=print):
    """Turn per-silenced-feature deltas into per-pair z / p-values under the chosen null.

    match_bg=True (default): accessibility-matched background -- bin response features by log(p0)
      (the model's baseline softmax share, the scale the delta lives on) and draw each
      candidate's background from its own bin. Controls for peak accessibility (like Signac/FigR),
      so high-accessibility promoter peaks are not scored as significant merely for being accessible.
    match_bg=False: random distal-feature background (per silenced feature). More sensitive, but its
      rankings are inflated by promoter/high-accessibility peaks; kept for benchmark parity.

    Returns a list of dict records: peak, gene, delta, zscore, pvalue, twosided_pvalue.
    """
    n_resp = D.shape[1]
    rng = np.random.default_rng(seed)
    all_resp = np.arange(n_resp)

    if match_bg:
        lp0 = np.log(np.maximum(p0, 1e-30))
        n_bins = int(min(n_bins, max(2, n_resp // (2 * n_bg))))
        qs = np.quantile(lp0, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        peak_bin = np.clip(np.digitize(lp0, qs[1:-1]), 0, n_bins - 1)
        bin_members = [np.where(peak_bin == b)[0] for b in range(n_bins)]
        log(f"  accessibility-matched null: {n_bins} log-p0 bins")

    records = []
    for row, silenced in enumerate(kept):
        d = D[row]
        cis_names = groups[silenced]
        cis_idx = np.array([resp_index[t] for t in cis_names], dtype=int)

        if not match_bg:
            pool = np.setdiff1d(all_resp, cis_idx, assume_unique=False)
            k = min(n_bg, pool.size)
            bg_idx = rng.choice(pool, size=k, replace=False) if k > 0 else all_resp
            mu_g, sd_g = float(d[bg_idx].mean()), float(d[bg_idx].std(ddof=1))
            if sd_g < EPS:
                sd_g = EPS

        for t, j in zip(cis_names, cis_idx):
            delta = float(d[j])
            if match_bg:
                b = int(peak_bin[j]); w = 0
                cand = bin_members[b]
                while cand.size - cis_idx.size < n_bg and w < n_bins:
                    w += 1; lo = max(0, b - w); hi = min(n_bins, b + w + 1)
                    cand = np.concatenate([bin_members[bb] for bb in range(lo, hi)])
                cand = cand[~np.isin(cand, cis_idx)]
                if cand.size > n_bg:
                    cand = rng.choice(cand, size=n_bg, replace=False)
                bgv = d[cand]
                mu, sd = float(bgv.mean()), float(bgv.std(ddof=1))
                if sd < EPS:
                    sd = EPS
            else:
                mu, sd = mu_g, sd_g
            z = (mu - delta) / sd                 # activating (delta<0) -> z>0
            p1 = float(norm.sf(z))                # 1-sided activating tail
            p2 = float(2.0 * norm.sf(abs(z)))
            peak, gene = (t, silenced) if direction == "gene2peak" else (silenced, t)
            records.append(dict(peak=peak, gene=gene, delta=delta,
                                zscore=z, pvalue=p1, twosided_pvalue=p2))
    return records


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values). Rank factor is m/i with i the
    ascending rank (1..m); the reverse-cumulative-min enforces monotonicity."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = p.argsort()
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def records_to_p2g(records):
    """Format scored records into the standard p2g DataFrame (peak coords parsed from the id)."""
    df = pd.DataFrame.from_records(records)
    df["fdr"] = bh_fdr(df["pvalue"].values)
    df["twosided_fdr"] = bh_fdr(df["twosided_pvalue"].values)
    df["score"] = -np.log10(df["pvalue"].clip(lower=1e-300))     # higher = more significant
    df["link_dir"] = -np.sign(df["delta"]).astype(int)          # +1 activating, -1 repressive
    parts = df["peak"].str.split("-", n=2, expand=True)
    df["peak_chr"], df["peak_start"], df["peak_end"] = parts[0], parts[1].astype(int), parts[2].astype(int)
    if not df["peak_chr"].str.startswith("chr").all():
        df["peak_chr"] = "chr" + df["peak_chr"].str.replace("^chr", "", regex=True)
        df["peak"] = df["peak_chr"] + "-" + df["peak_start"].astype(str) + "-" + df["peak_end"].astype(str)
    cols = ["peak", "peak_chr", "peak_start", "peak_end", "gene", "score", "delta",
            "zscore", "pvalue", "fdr", "twosided_pvalue", "twosided_fdr", "link_dir"]
    return df[cols]
