#!/usr/bin/env python3
"""Exactness test: mini-batching over cells must not change results.

Loads a trained model, learns representations ONCE, then computes deltas whole-batch
(cell_batch=n) vs a small cell_batch on the identical reps, and reports the max per-pair
difference in delta / zscore / score plus top-N rank overlap. Mini-batching over cells is a memory
layout change only, so differences should be float round-off (~1e-5) with identical ranking.

Usage (needs a trained multiDGD model + its export dir):
  python tests/test_exactness.py --input-dir <export> --save-dir <models> \
      --model-name dgd_x [--batch-correct] [--max-silenced 200] [--small-batch 512]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from multidgd_e2g import io as e2g_io          # noqa: E402
from multidgd_e2g import perturbation, scoring  # noqa: E402
from multidgd_e2g._adapter import MultiDGDAdapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--batch-correct", action="store_true")
    ap.add_argument("--direction", default="gene2peak")
    ap.add_argument("--max-silenced", type=int, default=200)
    ap.add_argument("--small-batch", type=int, default=512)
    ap.add_argument("--predict-epochs", type=int, default=20)
    a = ap.parse_args()

    adata, genes, peaks, pairs = e2g_io.read_export(a.input_dir)
    e2g_io.assign_train_val_split(adata)
    if not a.batch_correct:
        adata.obs.drop(columns=[c for c in adata.obs.columns if c.startswith("covariate_")],
                       inplace=True, errors="ignore")
    gene_row = {g: i for i, g in enumerate(genes)}
    peak_col = {p: i for i, p in enumerate(peaks)}
    groups, ai, ri = e2g_io.build_groups(pairs, gene_row, peak_col, a.direction,
                                         max_silenced=a.max_silenced)

    model = MultiDGDAdapter.load(adata, a.save_dir + os.sep, a.model_name)
    model.learn_test_representations(adata, n_epochs=a.predict_epochs)
    n = adata.n_obs

    Df, kf, p0f, _ = perturbation.compute_deltas(model, groups, ai, ri, a.direction, cell_batch=n + 1)
    Db, kb, p0b, _ = perturbation.compute_deltas(model, groups, ai, ri, a.direction, cell_batch=a.small_batch)
    assert kf == kb, "silenced sets differ"

    rf = pd.DataFrame(scoring.score_from_deltas(Df, kf, p0f, groups, ri, a.direction, match_bg=False)).set_index(["peak", "gene"])
    rb = pd.DataFrame(scoring.score_from_deltas(Db, kb, p0b, groups, ri, a.direction, match_bg=False)).set_index(["peak", "gene"]).reindex(rf.index)
    print(f"n_pairs = {len(rf)}", flush=True)
    for c in ["delta", "zscore", "pvalue"]:
        dd = (rf[c] - rb[c]).abs()
        print(f"per-pair |{c} diff|: max={dd.max():.3e} median={dd.median():.3e}", flush=True)
    sf = -np.log10(rf["pvalue"].clip(lower=1e-300)); sb = -np.log10(rb["pvalue"].clip(lower=1e-300))
    top = min(1000, len(rf))
    ov = len(set(sf.sort_values(ascending=False).head(top).index) &
             set(sb.sort_values(ascending=False).head(top).index))
    print(f"per-pair |score diff|: max={(sf - sb).abs().max():.3e} | top-{top} overlap {ov}/{top}", flush=True)
    print("PASS" if (sf - sb).abs().max() < 1e-2 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
