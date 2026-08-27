# multiDGD_E2G: Methods and Implementation

multiDGD_E2G performs genome-wide enhancer-gene linking by in-silico perturbation of a trained
[multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD) model. This document
describes what the perturbation measures, the scaling problem it solves, the cell mini-batching
engine and why it is exact, the calibrated scoring, and a performance/memory reference.

---

## 1. Background: in-silico perturbation on a generative model

multiDGD is an encoder-less deep generative model (Deep Generative Decoder) for paired RNA +
ATAC. Its structure: there is **no encoder** — each cell's latent representation `z_i` is a
free parameter optimized directly, and the latent prior is a **Gaussian Mixture Model** (which
captures cell-state clusters better than a single Gaussian and, with no encoder to feed, makes
the model data-efficient and able to handle genome-wide features without selection). A single
**shared decoder** (a small feed-forward neural network / MLP, plus a disentangled correction
latent for batch covariates) maps `z_i` to two per-modality output heads, each a **softmax over
that modality's features** (so each feature receives a share of the total); the
negative-binomial mean of a feature is `library_i × softmax_share`. There is no cross-cell
operation in the decoder — each cell's output depends only on its own latent.

An *in-silico perturbation* asks a counterfactual question: **if this feature were silenced,
what would the model predict for the others?** Concretely, set a chosen feature's counts to
zero, refit the per-cell latent to that counterfactual, decode, and read the change in every
partner-modality feature. A regulatory partner responds; an unrelated feature does not.
multiDGD introduced this as a proof of concept for gene-peak association. multiDGD_E2G runs it
at genome scale and turns the response into a calibrated link statistic.

---

## 2. The scaling problem

The response to a single silencing is a full decode: an `n_cells × n_peaks` softmax (plus the
`n_cells × n_genes` RNA head). For a large cell type that single tensor is tens of gigabytes
(e.g. 40,000 cells × 176,000 peaks × 4 bytes ≈ 28 GB), and a genome-wide analysis repeats the
perturbation for **every candidate feature** (each silenced in turn) — ~20,000 genes for `gene2peak`,
~100,000+ peaks for `peak2gene`. Materializing the whole count matrix and the whole decode on the
GPU for each such perturbation exhausts memory even on an 80 GB card.

Two observations make this tractable without changing the result:

1. The decoder is **per-cell**, so a decode can be computed on any subset of cells and the
   pieces combined.
2. The quantity of interest, the mean change over cells, is a **linear reduction** — a sum
   divided by a count — so chunk sums combine exactly.

---

## 3. The mini-batched perturbation engine (over cells)

`compute_deltas` (in `perturbation.py`) processes each silenced feature as follows, in cell chunks of
size `cell_batch`.

### 3.1 The silencing step

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

### 3.2 Why the mini-batching is exact

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

### 3.3 Memory

Peak GPU memory is dominated by the per-chunk decodes and scales as roughly
`2 × cell_batch × n_features × 4 bytes`, independent of the total cell count. Measured peaks:

| Cell type | cells | peaks | `cell_batch` | peak GPU |
|-----------|------:|------:|-------------:|---------:|
| Astrocytes | 6,255 | 102,482 | whole (6,255) | 9.52 GB |
| Astrocytes | 6,255 | 102,482 | 512 | **1.05 GB** (9.1×) |
| Oligodendrocytes | 34,784 | 107,651 | 8,192 | 12.64 GB |
| Excitatory | 39,884 | 176,001 | 8,192 | 23.36 GB |

The two largest cell types require an unbatched peak of ~80-85 GB — out of reach of an 80 GB
card — but fit a 48 GB card comfortably when batched, with headroom to raise `cell_batch` for
speed.

---

## 4. Calibrated association scoring

`score_from_deltas` (in `scoring.py`) turns the per-feature deltas into a per-pair significance
statistic. The design uses an empirical background null (random distal features by default;
optionally accessibility-matched), so the raw `delta` need not follow any parametric distribution.

For a silenced feature with cis targets and a background set of distal response features:

- **z-score** `z = (mean(delta_bg) − delta_cis) / sd(delta_bg)`. The sign convention makes an
  activating link (silencing an activator lowers the partner's predicted signal, so
  `delta_cis < 0`) yield a large **positive** `z`.
- **p-value** `pvalue = P(Z > z)` (1-sided, activating tail), with a 2-sided variant
  `2·P(Z > |z|)`.
- **FDR** Benjamini-Hochberg over all pairs.

**Background choices** (`--bg`):

- `random` — a random sample of `n_bg` non-cis response features per silenced feature. This also absorbs
  the global softmax mass shift that silencing induces across all features.
- `matched` — bin response features by `log(p0)`, where `p0` is the model's baseline predicted
  softmax share (the scale the delta lives on), and draw each candidate's background from its
  own bin. This controls for baseline accessibility.

Scoring is pure CPU and **null-agnostic with respect to the cached deltas** — changing `--bg`,
`n_bg`, the number of bins, or the sidedness re-runs only this step (Section 6).

---

## 5. Directions: gene2peak and peak2gene

The model is direction-agnostic; both directions reuse the same trained model and learned
representations — no retraining.

- **gene2peak** — silence a gene (RNA), read the peak (ATAC) response; background sampled over
  peaks. Silenced feature = a gene.
- **peak2gene** — silence a peak (ATAC), read the gene (RNA) response; background sampled over
  genes. Silenced feature = a peak.

Because runtime is `O(n_silenced × decode_cost)` (Section 7), `peak2gene` is proportionally
slower than `gene2peak` (peaks outnumber genes several-fold).

---

## 6. Delta caching

The expensive GPU output — the per-feature delta matrix `D` (kept silenced features × response features),
the baseline shares `p0`, and the feature names — is saved to
`deltas_lat{L}_{direction}.npz` after `compute_deltas`. It is null-agnostic, so any change to
the scoring null/sidedness re-scores from the cache on CPU (`--reuse-deltas`) in minutes, with
no model load and no GPU. The cache is keyed by latent dimension and direction so different
model configurations do not collide.

---

## 7. Performance and memory reference

Runtime is `O(n_silenced × decode_cost)`, where `decode_cost ∝ n_cells × n_peaks`. The
perturbation is a **sequential loop over silenced features** (each is a separate counterfactual refit), so
the GPU accelerates each decode but cannot parallelize across features. `--cell-batch` trades
memory for speed: fewer chunks mean fewer decodes.

Measured on 48 GB cards (A40/L40), `gene2peak`, `cell_batch = 8192`, batch-corrected, latent 20:

| Cell type | cells | peaks | silenced | training | delta pass | total | peak GPU |
|-----------|------:|------:|--------:|---------:|-----------:|------:|---------:|
| Oligodendrocytes | 34,784 | 107,651 | 20,204 | 2.6 h | 7.5 h | 10.5 h | 12.64 GB |
| Excitatory | 39,884 | 176,001 | 20,263 | 5.1 h | 15.4 h | 21.6 h | 23.36 GB |

Training and representation learning are multiDGD's own code (unaffected by batching); the delta
pass is the perturbation engine. Two levers reduce the delta pass on large cell types (both
exact): a **larger `cell_batch`** (fewer chunks) where GPU memory allows, and holding the
baseline decode resident to avoid recomputing it per chunk (feasible on large-memory cards).
Smaller cell types (`n_cells ≤ cell_batch`) run in a single chunk and finish in minutes to an
hour.
