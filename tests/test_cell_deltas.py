#!/usr/bin/env python3
"""Consistency test for compute_cell_deltas.

The per-cell delta matrix from `compute_cell_deltas` (one silenced feature) must average, over the same
detected cells, to the aggregated row that `compute_deltas` produces for that silenced feature. This
checks the per-cell engine is the un-summed version of the main engine.

Usage (needs a trained multiDGD model + its export dir):
  python tests/test_cell_deltas.py --input-dir <export> --save-dir <models> \
      --model-name dgd_x [--direction gene2peak] [--predict-epochs 20]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from multidgd_e2g import io as e2g_io          # noqa: E402
from multidgd_e2g import perturbation           # noqa: E402
from multidgd_e2g._adapter import MultiDGDAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--direction", default="gene2peak")
    ap.add_argument("--predict-epochs", type=int, default=20)
    ap.add_argument("--cell-batch", type=int, default=1024)
    a = ap.parse_args()

    adata, genes, peaks, pairs = e2g_io.read_export(a.input_dir)
    e2g_io.assign_train_val_split(adata)
    # keep any covariate_ columns so the loaded model's correction factors line up
    gene_row = {g: i for i, g in enumerate(genes)}
    peak_col = {p: i for i, p in enumerate(peaks)}
    groups, ai, ri = e2g_io.build_groups(pairs, gene_row, peak_col, a.direction)

    model = MultiDGDAdapter.load(adata, a.save_dir + os.sep, a.model_name)
    model.learn_test_representations(adata, n_epochs=a.predict_epochs)

    D, kept, p0, resp_names = perturbation.compute_deltas(
        model, groups, ai, ri, a.direction, cell_batch=a.cell_batch)
    silenced = kept[0]
    targets = groups[silenced]
    resp_cols = [ri[t] for t in targets]
    agg_row = D[kept.index(silenced), resp_cols]     # aggregated mean over detected cells

    delta_cells, cell_idx = perturbation.compute_cell_deltas(
        model, silenced, ai, resp_cols, a.direction, cell_batch=a.cell_batch)
    per_cell_mean = delta_cells.mean(axis=0)

    diff = np.abs(per_cell_mean - agg_row)
    print(f"silenced={silenced} detected_cells={len(cell_idx)} targets={len(targets)}", flush=True)
    print(f"per-cell mean vs aggregated: max|diff|={diff.max():.3e} median={np.median(diff):.3e}", flush=True)
    print("PASS" if diff.max() < 1e-4 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
