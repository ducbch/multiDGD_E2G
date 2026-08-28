# multiDGD_E2G: Methods and Implementation

multiDGD_E2G performs genome-wide enhancer-gene linking by in-silico perturbation of a trained
[multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD) model. This document
describes what the perturbation measures, how it is scaled genome-wide with an exact cell
mini-batching engine, the calibrated scoring, and a performance reference.

---

## 1. Background: in-silico perturbation on a generative model

multiDGD is an encoder-less deep generative model (Deep Generative Decoder) for paired RNA +
ATAC. Its structure: there is **no encoder** — each cell's latent representation `z_i` is a
free parameter optimized directly, and the latent prior is a **Gaussian Mixture Model** (which
captures cell-state clusters better than a single Gaussian and, with no encoder to feed, makes
the model data-efficient and able to handle genome-wide features without selection). A single
**shared decoder** (a small feed-forward neural network / MLP, plus a disentangled correction
latent for batch covariates) maps `z_i` to two per-modality outputs, one for RNA and one for ATAC.
Each output is a **fraction per feature that sums to 1 across the modality** (a softmax): the
model's predicted share of that cell's signal for each gene or peak — call it the feature's
**predicted accessibility** (peaks) or expression fraction (genes). A feature's expected count is
`library_i × this fraction`. There is no cross-cell operation in the decoder — each cell's output
depends only on its own latent.

An *in-silico perturbation* asks a counterfactual question: **if this feature were silenced,
what would the model predict for the others?** Concretely, set a chosen feature's counts to
zero, refit the per-cell latent to that counterfactual, decode, and read the change in every
partner-modality feature. A regulatory partner responds; an unrelated feature does not.
multiDGD introduced this as a proof of concept for gene-peak association. multiDGD_E2G runs it
at genome scale and turns the response into a calibrated link statistic.

---

## 2. Scaling the perturbation genome-wide

The response to a single silencing is a full decode: an `n_cells × n_peaks` output matrix (plus the
`n_cells × n_genes` RNA head). For a large cell type that single tensor is tens of gigabytes
(e.g. 40,000 cells × 176,000 peaks × 4 bytes ≈ 28 GB), and a genome-wide analysis repeats the
perturbation for **every candidate feature** (each silenced in turn) — ~20,000 genes for `gene2peak`,
~100,000+ peaks for `peak2gene`. Materializing the whole count matrix and the whole decode on the
GPU for each such perturbation exhausts memory even on an 80 GB card.

Two properties make this tractable without changing the result: the decoder is per-cell (§1), so a
decode can run on any subset of cells and be recombined; and the quantity of interest — the mean
change over cells — is a **linear reduction** (a sum over a count), so chunk sums combine exactly.

The engine (`compute_deltas` in `perturbation.py`) processes each silenced feature in cell chunks
of size `cell_batch`.

### 2.1 The silencing step

For a silenced feature `f`:

- **Counterfactual target.** Only the silenced modality's counts are needed, and only the
  silenced feature's own column is read by the loss — so per chunk we move that modality's slice to the
  GPU and zero column `f`. The full count matrix never leaves CPU.
- **Representation (rep) refit.** One Adam step on the per-cell latent representation `z`
  minimizes the reconstruction loss of feature `f` under the counterfactual. Chunked, this is
  **gradient accumulation**: `zero_grad`
  once, back-propagate each chunk's loss into `z.grad`, then a single `step`. Because each
  cell's gradient depends only on its own latent and the chunks are disjoint, the accumulated
  gradient equals the full-batch gradient exactly.
- **Delta.** Decode the perturbed latent and the baseline latent per chunk, take the difference
  on the response modality, and accumulate `sum_delta` and `count` over detected cells (cells
  where the silenced feature is nonzero). After the last chunk, `delta = sum_delta / count`.
- **Restore.** The latent is reset to the saved baseline `z0` before the next feature.

### 2.2 Why the mini-batching is exact

The chunked computation is not an approximation. The decoder has **no BatchNorm or other
cross-cell op**, so a cell's decode is identical whether it is computed alone or in a batch; the
delta mean is a linear reduction combined exactly across chunks; and the rep refit accumulates
gradients before one optimizer step (== the full-batch gradient), rather than taking a step per
chunk. The only difference from a whole-matrix computation is float32 summation order.

Verified on two cell types (whole vs small `cell_batch`, scored under the identical null):

| Cell type | pairs | chunks | max \|zscore diff\| | max \|score diff\| | top-1000 rank overlap |
|-----------|------:|-------:|--------------------:|-------------------:|:---------------------:|
| Microglia (3,179 cells) | 5,604 | 7 | 1.2e-4 | 2.1e-5 | 1000 / 1000 |
| Astrocytes (6,255 cells) | 8,491 | 13 | 8.5e-4 | 1.3e-4 | 1000 / 1000 |

Per-pair score differences are float round-off (medians ~1e-6); rankings are identical. The
difference grows gently with chunk count (more summation terms) but stays negligible.

### 2.3 Memory

Peak GPU memory is dominated by the per-chunk decodes and scales as roughly
`2 × cell_batch × n_features × 4 bytes`, independent of the total cell count — so batching is what
makes large cell types fit. Measured peak GPU, whole-matrix vs batched:

| Cell type | cells | peaks | whole-matrix | batched |
|-----------|------:|------:|--------------|---------|
| Astrocytes | 6,255 | 102,482 | 9.52 GB | 1.05 GB (`cell_batch=512`, 9.1× smaller) |
| Oligodendrocytes | 34,784 | 107,651 | ~80 GB — **exceeds an A100 (80 GB)** | 12.64 GB — fits an A40/L40 (48 GB) |
| Excitatory | 39,884 | 176,001 | ~85 GB — **exceeds an A100 (80 GB)** | 23.36 GB — fits an A40/L40 (48 GB) |

The two largest cell types cannot run whole-matrix even on an 80 GB card, but batch comfortably onto
a 48 GB card, with headroom to raise `cell_batch` for speed.

---

## 3. Calibrated association scoring

`score_from_deltas` (in `scoring.py`) turns the per-feature deltas into a per-pair significance
statistic against an empirical background null (selected with `--bg`; see below), so the raw
`delta` need not follow any parametric distribution.

For a silenced feature with cis targets and a background set of distal response features:

- **z-score** `z = (mean(delta_bg) − delta_cis) / sd(delta_bg)`. The sign convention makes an
  activating link (silencing an activator lowers the partner's predicted signal, so
  `delta_cis < 0`) yield a large **positive** `z`.
- **p-value** `pvalue = P(Z > z)` (1-sided, activating tail), with a 2-sided variant
  `2·P(Z > |z|)`.
- **FDR** Benjamini-Hochberg over all pairs.

The two `--bg` backgrounds — `matched` (default) and `random` — and why the default matters are
covered next. Scoring is pure CPU and **null-agnostic with respect to the cached deltas**: changing
`--bg`, `n_bg`, the number of bins, or the sidedness re-runs only this step (Section 5).

### Accessibility bias and the matched null

**The bias.** The raw perturbation effect `ΔX̂` scales with a peak's *baseline accessibility*.
Silencing a gene moves its high-accessibility, promoter-proximal peaks the most, so the raw
`|ΔX̂|` is largest at the TSS and decays to roughly a third within ~2 kb — a ranking dominated by
promoters and other highly accessible peaks. This is a known limitation of the underlying method: multiDGD's
paper reports that "ΔX̂ is not independent of mean availability" (Schuster et al., *Nat Commun*
2024, Supplementary Fig. 21).

**Why it happens.** As in Section 1, the ATAC output is a softmax, so for cell *i* peak *j*'s
predicted accessibility is a fraction `ŷ_ij ∈ (0,1)` with `Σ_j ŷ_ij = 1`. The perturbation records
the mean change in that fraction,

    ΔX̂_gj = mean_i ( ŷ^KO_ij − ŷ^0_ij ),

and because the fractions must always sum to 1, the change redistributed by silencing is
proportionally larger where the baseline fraction is larger — so already-accessible peaks carry
larger `|ΔX̂|` whether or not there is a true regulatory link.

**`p0` — the matching variable (a *predicted* quantity, not empirical ATAC).** The right scale to
correct on is therefore the model's own predicted baseline accessibility. `compute_deltas` returns

    p0_j = (1/N) · Σ_i ŷ^0_ij ,

the mean over all `N` cells of peak `j`'s **baseline (unperturbed) predicted accessibility**,
obtained by decoding each cell's baseline latent `z^0`. So `p0` is a model prediction — a fraction
on the order of `1/n_peaks` (~1e-4) — on the *same* scale as `ΔX̂`. It is **not** the empirical
fraction of cells with an ATAC count in the peak; matching on the model's predicted accessibility
is what cancels the accessibility-driven magnitude baked into `ΔX̂`.

**The matched null (`--bg matched`, the default).** Scoring bins peaks by `log p0` into
`B = min(25, n_resp / (2·n_bg))` quantile bins. For a candidate (silenced feature `g`, cis peak `j`):

1. draw the background from peaks in `j`'s own `log p0` bin (excluding `g`'s cis peaks; widen to
   neighbouring bins if fewer than `n_bg` remain) → background deltas `d_bg`;
2. `z_gj = ( mean(d_bg) − ΔX̂_gj ) / sd(d_bg)`, and `pvalue = P(Z > z_gj)` (one-sided).

Because `j` is compared only against peaks of **equal predicted accessibility**, a large `|ΔX̂|`
that is merely a consequence of high `p0` no longer reads as significant. Empirically this makes the
association score `|z|` **flat with distance to the TSS** — the raw `|ΔX̂|` decay within 2 kb
disappears — so links are no longer dominated by promoters/abundant peaks. The `--bg random` null
skips this correction (background = random distal peaks) and is kept only for benchmark parity.

---

## 4. Directions: gene2peak and peak2gene

The model is direction-agnostic; both directions reuse the same trained model and learned
representations — no retraining.

- **gene2peak** — silence a gene (RNA), read the peak (ATAC) response; background sampled over
  peaks. Silenced feature = a gene.
- **peak2gene** — silence a peak (ATAC), read the gene (RNA) response; background sampled over
  genes. Silenced feature = a peak.

`peak2gene` runs proportionally slower than `gene2peak` — it silences every peak rather than every
gene, and peaks outnumber genes several-fold (see runtime in Section 6).

---

## 5. Delta caching

The expensive GPU output — the per-feature delta matrix `D` (kept silenced features × response features),
the baseline shares `p0`, and the feature names — is saved to
`deltas_lat{L}_{direction}.npz` after `compute_deltas`. It is null-agnostic, so any change to
the scoring null/sidedness re-scores from the cache on CPU (`--reuse-deltas`) in minutes, with
no model load and no GPU. The cache is keyed by latent dimension and direction so different
model configurations do not collide.

---

## 6. Performance reference

Runtime is `O(n_silenced × decode_cost)`, where `decode_cost ∝ n_cells × n_peaks`. The
perturbation is a **sequential loop over silenced features** (each is a separate counterfactual refit), so
the GPU accelerates each decode but cannot parallelize across features. `--cell-batch` trades
memory for speed: fewer chunks mean fewer decodes.

Measured on 48 GB cards (A40/L40), `gene2peak`, `cell_batch = 8192`, batch-corrected, latent 20:

| Cell type | cells | peaks | silenced | training | delta pass | total |
|-----------|------:|------:|--------:|---------:|-----------:|------:|
| Oligodendrocytes | 34,784 | 107,651 | 20,204 | 2.6 h | 7.5 h | 10.5 h |
| Excitatory | 39,884 | 176,001 | 20,263 | 5.1 h | 15.4 h | 21.6 h |

Training and representation learning are multiDGD's own code (unaffected by batching); the delta
pass is the perturbation engine. Two levers reduce the delta pass on large cell types (both
exact): a **larger `cell_batch`** (fewer chunks) where GPU memory allows, and holding the
baseline decode resident to avoid recomputing it per chunk (feasible on large-memory cards).
Smaller cell types (`n_cells ≤ cell_batch`) run in a single chunk.
