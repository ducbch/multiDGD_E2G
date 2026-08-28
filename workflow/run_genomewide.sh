#!/usr/bin/env bash
# run_genomewide.sh -- production genome-wide enhancer-gene run via the multidgd-e2g CLI.
# Trains multiDGD on a full (genome-wide) export, runs the gene2peak perturbation over every
# gene, and scores into a link table + a saved model. This is a heavy GPU job (hours);
# visualize_run.ipynb then just LOADS the saved model + link table it produces.
#
# Requires multiDGD_E2G installed and its environment activated (see ../README.md).
#
# Usage (edit the paths, then run directly or submit to your scheduler):
#   bash run_genomewide.sh <export_dir> <save_dir> <out.tsv>
# Submit to SLURM, e.g.:
#   sbatch --account=<acct> --partition=<gpu_partition> --gres=gpu:1 --mem=192G \
#          --time=48:00:00 --cpus-per-task=8 --wrap "bash run_genomewide.sh <export> <save> <out>"
#
# The export dir is a genome-wide flat export (all genes/peaks + candidate pairs); build it
# from your object with `multidgd-e2g prep` (.h5ad/.h5mu) or scripts/seurat_to_export.R.
# Two-sample data -> --batch-correct on (obs 'covariate_sample'). Defaults (500 epochs, 20
# predict epochs, n_bg 500, seed 0, cell_batch 8192) match a production run.

set -eo pipefail
cd "$(dirname "$0")"

INPUT_DIR=${1:?usage: run_genomewide.sh <export_dir> <save_dir> <out.tsv>}
SAVE_DIR=${2:?usage: run_genomewide.sh <export_dir> <save_dir> <out.tsv>}
OUT=${3:?usage: run_genomewide.sh <export_dir> <save_dir> <out.tsv>}

multidgd-e2g link \
  --train \
  --input-dir  "$INPUT_DIR" \
  --save-dir   "$SAVE_DIR" \
  --model-name dgd_cd14_e2g \
  --direction  gene2peak \
  --bg         matched \
  --batch-correct \
  --latent-dim 20 \
  --cell-batch 8192 \
  --out        "$OUT"

echo "Done -> $SAVE_DIR (model) + $OUT (links)"
