# multiDGD_E2G

**Genome-wide Enhancer-to-Gene linking via multiDGD in-silico perturbation**

multiDGD_E2G links cis-regulatory elements (ATAC-seq peaks) to their target genes from
single-cell multimodal (RNA + ATAC) data, using in-silico perturbation of a trained
[multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD) generative model. It
silences a feature, refits the per-cell latent, decodes, and measures the predicted change in
the partner modality — then scores that change into a calibrated link statistic. multiDGD
introduced this perturbation idea as a **proof of concept** (per-gene, small scale); multiDGD_E2G
develops it into a **scalable, genome-wide** tool by running the perturbation in cell chunks so
GPU memory is bounded by the chunk size rather than the cell count.

Method background: Schuster et al., *Nat Commun* 15, 10031 (2024),
[doi:10.1038/s41467-024-53340-z](https://doi.org/10.1038/s41467-024-53340-z).

---

## Why multiDGD_E2G?

A trained multiDGD model is an encoder-less deep generative model: each cell's latent is a free
parameter under a Gaussian-Mixture prior, and a shared decoder maps it to per-modality softmax
outputs (negative-binomial mean = library × softmax share). Silencing a feature and reading the
model's predicted response in the other modality is a natural way to propose regulatory links.

Running this genome-wide is memory-bound: each perturbation decodes the model's output over all
features for all cells (an `n_cells × n_peaks` softmax), repeated across tens of thousands of perturbations —
one per **silenced feature** (every gene for `gene2peak`, every peak
for `peak2gene`). Materializing that whole decode exceeds GPU memory at realistic cell counts.

multiDGD_E2G runs the perturbation in **cell chunks**: the count matrix stays on CPU, each decode
is computed per chunk, and the per-cell mean change is accumulated across chunks. Peak GPU memory
then scales with the chunk size rather than the cell count (see [Scalability](#scalability)), and
the chunked result matches the whole-matrix result. The model, decoder, and training are
multiDGD's; multiDGD_E2G adds the scalable perturbation, a calibrated significance score, and
peak-gene I/O, with every multiDGD-internal call isolated in one adapter module.

---

## Scalability

The original whole-matrix perturbation materializes the full decode on the GPU, which exceeds an
80 GB A100 for large cell types. Mini-batching over cells (processing them in chunks of `cell_batch`,
one chunk at a time) bounds the footprint to a single chunk, so the
same analyses run on a 48 GB card. Measured on brain multiome data (`gene2peak`, latent 20):

| Cell type | cells | peaks | whole-matrix | cell-batched (`cell_batch=8192`) |
|-----------|------:|------:|--------------|----------------------------------|
| Oligodendrocytes | 34,784 | 107,651 | ~80 GB — exceeds A100 (80 GB) | 12.6 GB — runs on A40/L40 (48 GB) |
| Excitatory | 39,884 | 176,001 | ~85 GB — exceeds A100 (80 GB) | 23.4 GB — runs on A40/L40 (48 GB) |

The chunked result is numerically equivalent to the whole-matrix result; a runtime reference and
the equivalence details are in [DEEP_README.md](DEEP_README.md).

---

## Installation

### Prerequisites

- Python >= 3.8 (tested on 3.10)
- A working [multiDGD](https://github.com/Center-for-Health-Data-Science/multiDGD) install
  (the model backend). multiDGD pins a 2022-era stack (torch 1.13.1 / numpy 1.23.5 /
  anndata 0.8.0), so it is cleanest in a dedicated environment.
- A GPU is strongly recommended (the perturbation is compute-heavy).

### Install

```bash
# dedicated environment matching multiDGD's stack
conda create -n multidgd-e2g python=3.10 -y && conda activate multidgd-e2g
pip install multiDGD@git+https://github.com/Center-for-Health-Data-Science/multiDGD

# multiDGD_E2G
pip install git+https://github.com/ducbch/multiDGD_E2G

# ...or for a local/editable checkout:
#   git clone https://github.com/ducbch/multiDGD_E2G && pip install -e multiDGD_E2G
```

### Verify

```bash
multidgd-e2g link --help
python -c "import multidgd_e2g; print(multidgd_e2g.__version__)"
```

---

## Quick Start

A full run is: prepare inputs → train the multiDGD model → perturb & score. The `link` command
does the last two in one step, and it does **not** auto-train — pass `--train` the first time
(fits the model from scratch), then `--load` to reuse that model.

See [`examples/`](examples/EXAMPLES_README.md) for a small, self-contained input-conversion
example (a two-sample CD14 monocyte subset in AnnData/MuData/Seurat form) plus a quick CLI smoke
test, and [`workflow/`](workflow/WORKFLOW_README.md) for the genome-wide production run
(`run_genomewide.sh`, GPU) and `visualize_run.ipynb`, which loads a run's saved model + links and
makes figures.

### 1. Prepare your data

multiDGD_E2G reads a small **export directory** (a plain, language-agnostic bridge from
Seurat/AnnData):

```
export_dir/
  rna_counts.mtx      # genes x cells (MatrixMarket)
  atac_counts.mtx     # peaks x cells (MatrixMarket)
  rna_features.txt    # gene names (row order of rna_counts)
  atac_features.txt   # peak ids "chr-start-end" (row order of atac_counts)
  barcodes.txt        # cell barcodes (column order)
  obs.csv             # per-cell: barcode, observable (cell type), covariate_sample (optional)
  pairs.csv           # candidate gene,peak pairs
```

You do not have to write these files by hand — convert your existing object to the export
directory with one of the provided helpers (each also generates `pairs.csv` when given a gene
model, or copies a `--pairs`/`--gtf` you supply):

```bash
# AnnData (.h5ad, var has modality/feature_types)
multidgd-e2g prep --adata data.h5ad  --out export_dir --observable celltype --gtf genes.gtf

# MuData (.h5mu, RNA + ATAC modalities)
multidgd-e2g prep --mudata data.h5mu --out export_dir --observable celltype --gtf genes.gtf

# Seurat (.rds), via the bundled R script
Rscript scripts/seurat_to_export.R --rds obj.rds --out export_dir \
  --celltype-col celltype --genebed genes.bed
```

Use `--covariate`/`--sample-col` to carry a batch column (written as `covariate_sample`), and
`--pairs pairs.csv` instead of `--gtf`/`--genebed` if you already have candidate pairs. From
Python you can also build the pairs directly from coordinates:

```python
from multidgd_e2g import io
pairs = io.candidate_pairs_by_distance(gene_tss, peaks, distance=250_000)
```

### 2. Train the model and link (first run)

One command trains multiDGD from scratch, learns per-cell representations, runs the perturbation,
and writes the link table:

```bash
multidgd-e2g link \
  --train \                       # fit a new model (first run); --load reuses it afterwards
  --input-dir  export_dir \
  --save-dir   model_dir \        # model checkpoint + delta cache are written here
  --model-name dgd_mydata \
  --direction  gene2peak \        # silence gene -> read peaks (or peak2gene)
  --bg         matched \          # null background: accessibility-matched (default) or random
  --cell-batch 8192 \             # cells per decode chunk (set as large as the GPU allows)
  --out        mydata_e2g.tsv
```

### 3. Reuse the trained model (`--load`)

Later runs reuse the checkpoint — no retraining — e.g. the other direction:

```bash
multidgd-e2g link --load --input-dir export_dir --save-dir model_dir --model-name dgd_mydata \
  --direction peak2gene --out mydata_e2g_peak2gene.tsv
```

### 4. Re-score under a different null — no GPU

Deltas are cached, so changing the null/sidedness re-runs only the (CPU) scoring:

```bash
multidgd-e2g link --reuse-deltas --input-dir export_dir --save-dir model_dir \
  --model-name dgd_mydata --bg random --out mydata_e2g_random.tsv
```

---

## Python API

The same pipeline as callable functions — an **alternative** to the CLI (steps 2-4), for notebooks
or custom pipelines where you want the intermediate objects (`D`, `records`) in hand. Swap
`io.read_export` for `io.build_anndata(...)` to work from your own matrices without an export directory.

```python
from multidgd_e2g import MultiDGDAdapter, compute_deltas, score_from_deltas, records_to_p2g
from multidgd_e2g import io

adata, genes, peaks, pairs = io.read_export("export_dir")
io.assign_train_val_split(adata)                       # multiDGD needs a train/validation split

model = MultiDGDAdapter.train(adata, "model_dir/", "dgd_mydata", latent_dim=20)  # or .load(...)
model.learn_test_representations(adata)                # learn per-cell latents over all cells

gene_row = {g: i for i, g in enumerate(genes)}
peak_col = {p: i for i, p in enumerate(peaks)}
groups, si, ri = io.build_groups(pairs, gene_row, peak_col, "gene2peak")

D, kept, p0, _ = compute_deltas(model, groups, si, ri, "gene2peak", cell_batch=8192)
records = score_from_deltas(D, kept, p0, groups, ri, "gene2peak", match_bg=False)
records_to_p2g(records).to_csv("mydata_e2g.tsv", sep="\t", index=False)
```

---

## Options

### `--direction` — which feature is perturbed

| Direction | What it does | Silenced |
|-----------|--------------|---------|
| `gene2peak` (default) | Silence a gene, read the predicted change in peaks. | genes |
| `peak2gene` | Silence a peak, read the predicted change in genes. | peaks |

Both reuse the *same* trained model (the model is direction-agnostic). `peak2gene` has many
more silenced features (peaks ≫ genes), so it is proportionally slower.

### `--bg` — the null background

| Background | Description |
|------------|-------------|
| `matched` (default) | Background drawn from the candidate's own baseline-accessibility bin (the model's softmax share), controlling for peak accessibility — the same control Signac/FigR apply. Prevents high-accessibility promoter peaks from scoring as significant merely for being accessible. |
| `random` | Each candidate's delta is z-scored against a random sample of distal (non-cis) response features. More sensitive, but its rankings are inflated by promoter/high-accessibility peaks; use for benchmark parity with parametric methods, not as the discovery default. |

### `--cell-batch` — memory/speed knob

Cells per decode chunk: larger uses more GPU memory but runs faster (fewer chunks); the result
is identical regardless of the value. Cell types with `n_cells ≤ cell_batch` run in one chunk.

### `--load` / `--train`, `--reuse-deltas`

`--load` reuses a saved model checkpoint; `--train` fits a new one. `--reuse-deltas` skips the
model entirely and re-scores from the cached deltas on CPU.

---

## Output

`link` writes a tab-separated table with one row per candidate pair:

| Column | Description |
|--------|-------------|
| `peak` | Peak id (`chr-start-end`) |
| `peak_chr`, `peak_start`, `peak_end` | Parsed peak coordinates |
| `gene` | Gene name |
| `score` | `-log10(pvalue)` — higher = more significant (the ranking value) |
| `delta` | Mean predicted change of the target under silencing (raw model output) |
| `zscore` | Standardized delta vs the background null |
| `pvalue` | 1-sided (activating-tail) p-value |
| `fdr` | Benjamini-Hochberg FDR of `pvalue` |
| `twosided_pvalue`, `twosided_fdr` | Direction-agnostic variant |
| `link_dir` | `+1` activating / `-1` repressive (sign of the effect) |

---

## In-silico perturbation and scoring (summary)

- **Perturbation.** For each silenced feature: zero its counts, take one gradient step on the per-cell
  latent to fit the counterfactual, decode, and record the mean predicted change (`delta`) in
  every partner-modality feature — over cells where that feature is detected. Run in cell chunks
  so memory is bounded; the chunked result equals the whole-matrix result up to float round-off.
- **Scoring.** An empirical background null: z-score each cis target's `delta` against the
  candidate's own accessibility bin (`--bg matched`, the default) or a background of random
  distal features (`--bg random`). `pvalue` is 1-sided on the
  activating tail (silencing an activator lowers the partner's predicted signal), with a
  2-sided variant and BH FDR. No distributional assumption on the raw `delta`.

Full details, the exactness argument, and a performance/memory reference are in
[DEEP_README.md](DEEP_README.md).

---

## Credits

multiDGD_E2G is developed by **Duc Nguyen** at the Lee Lab, Boston Children's Hospital.

The generative model and the in-silico perturbation proof of concept are from multiDGD:

> **Schuster, V.**, Dann, E., Krogh, A., Teichmann, S. A. (2024). multiDGD: A versatile deep
> generative model for multi-omics data. *Nature Communications* 15, 10031.
> [doi:10.1038/s41467-024-53340-z](https://doi.org/10.1038/s41467-024-53340-z)

multiDGD source code: https://github.com/Center-for-Health-Data-Science/multiDGD

---

## License

MIT
