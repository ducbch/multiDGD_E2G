# Workflow (genome-wide production run)

The real analysis: train multiDGD genome-wide, run the gene2peak perturbation over every gene,
score into a link table, then visualize. This is a heavy GPU job (hours) and its outputs (model,
delta cache, link table) are large — they live **outside** the repo, not in git.

```
workflow/
  run_genomewide.sh     # train + perturb + score, genome-wide, via the multidgd-e2g CLI
  visualize_run.ipynb   # load the saved model + links and make figures (no retraining)
```

Why genome-wide (not the small `examples/` subset): the latent, the softmax over peaks, and the
scoring background null must all span the whole feature set. A regional subset confines all three
to one locus and does not give valid links.

## 1. Build a genome-wide export

A flat export directory with all genes/peaks + candidate pairs (peaks within ±250 kb of each
gene TSS), plus `covariate_sample` for batch correction:

```bash
# from AnnData / MuData:
multidgd-e2g prep --adata full.h5ad --out export_dir --observable celltype --covariate sample --gtf genes.gtf
# or from Seurat:
Rscript ../scripts/seurat_to_export.R --rds full.rds --out export_dir \
  --celltype-col celltype --sample-col sample --genebed genes.bed
```

## 2. Run (GPU, hours)

```bash
sbatch --account=<acct> --partition=<gpu_partition> --gres=gpu:1 --mem=192G \
       --time=48:00:00 --cpus-per-task=8 \
       --wrap "bash run_genomewide.sh <export_dir> <run_dir/model> <run_dir/links.tsv>"
```

Writes the trained model to `<run_dir/model>` and the link table to `<run_dir/links.tsv>`
(defaults: 500 epochs, 20 predict epochs, n_bg 500, seed 0, cell_batch 8192, `--batch-correct`,
`gene2peak`).

## 3. Visualize

Open `visualize_run.ipynb`, set the path variables at the top (`EXPORT_DIR`, `MODEL_DIR`,
`MODEL_NAME`, `LINKS`, `GTF`) to your run output, and run it. It **loads** the saved model +
links (no retraining) and renders: training/latent diagnostics, the covariate (sample) latent,
the volcano (one- vs two-sided), a marker-gene locus view, and a per-cell heatmap. The heavy
train/perturbation steps are shown as commented reference cells.

## Notes

- The saved model is ~150 MB and the link table can be large; keep them outside git.
- The run uses the accessibility-matched null (`--bg matched`, the default). For the random-distal
  null (benchmark parity) or the reverse direction, re-run with `--bg random` or
  `--direction peak2gene` (re-scoring only, `--reuse-deltas`, no GPU).
