"""Command-line interface for multiDGD_E2G.

    multidgd-e2g prep (--adata a.h5ad | --mudata a.h5mu) --out <export> \
        [--observable <col>] [--covariate <col>] [--gtf g.gtf | --pairs p.csv]

    multidgd-e2g link --input-dir <export> --save-dir <models> --model-name <name> \
        [--train | --load] [--direction gene2peak|peak2gene] [--bg matched|random] \
        [--cell-batch 8192] [--reuse-deltas] --out <links.tsv>

`prep` converts a .h5ad/.h5mu into the flat export directory (the Seurat path is
scripts/seurat_to_export.R). `link` loads/trains a multiDGD model, learns per-cell
representations over all cells, runs the in-silico perturbation (silence each silenced feature,
mini-batched over cells), and scores the deltas into a calibrated enhancer-gene link table. Deltas are
cached so a null/sidedness change re-scores on CPU (--reuse-deltas) with no GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import io as e2g_io
from . import scoring


def _log(msg):
    print(msg, flush=True)


def link(args):
    from . import perturbation
    from ._adapter import MultiDGDAdapter
    adata, genes, peaks, pairs = e2g_io.read_export(args.input_dir)
    e2g_io.assign_train_val_split(adata, train_fraction=args.train_fraction, seed=args.seed)
    _log(f"{adata.n_obs} cells | {len(genes)} genes | {len(peaks)} peaks")

    gene_row = {g: i for i, g in enumerate(genes)}
    peak_col = {p: i for i, p in enumerate(peaks)}
    cov_cols = [c for c in adata.obs.columns if c.startswith("covariate_")]
    if not args.batch_correct:
        adata.obs.drop(columns=cov_cols, inplace=True, errors="ignore")
        cov_cols = []

    if pairs is None:
        _log("ERROR: no pairs.csv in the export; supply candidate pairs "
             "(gene,peak) or generate with io.candidate_pairs_by_distance.")
        sys.exit(1)
    groups, silenced_index, resp_index = e2g_io.build_groups(
        pairs, gene_row, peak_col, args.direction, max_silenced=args.max_silenced)

    os.makedirs(args.save_dir, exist_ok=True)
    cache = os.path.join(args.save_dir, f"deltas_lat{args.latent_dim}_{args.direction}.npz")

    if args.reuse_deltas:
        if not os.path.exists(cache):
            _log(f"ERROR: --reuse-deltas but no cache at {cache}"); sys.exit(1)
        _log(f"Reusing cached deltas (no model/GPU): {cache}")
        cz = np.load(cache, allow_pickle=True)
        D, kept, p0, resp_names = cz["D"], list(cz["kept"]), cz["p0"], list(cz["resp_names"])
        cur = [None] * len(resp_index)
        for nm, j in resp_index.items():
            cur[j] = nm
        if list(resp_names) != cur:
            _log("ERROR: cached delta feature order != current candidates; recompute."); sys.exit(1)
        keep_rows = [i for i, a in enumerate(kept) if a in groups]
        D, kept = D[keep_rows], [kept[i] for i in keep_rows]
    else:
        n_ct = adata.obs["observable"].nunique() if "observable" in adata.obs else 1
        ckpt = os.path.join(args.save_dir, args.model_name + "_hyperparameters.json")
        if args.load and not os.path.exists(ckpt):
            _log(f"ERROR: --load but no trained model at {ckpt}. "
                 f"Run once with --train to fit it (link does not auto-train)."); sys.exit(1)
        if args.load:
            _log("Loading cached model ...")
            model = MultiDGDAdapter.load(adata, args.save_dir + os.sep, args.model_name)
        else:
            _log(f"Training DGD (latent={args.latent_dim}) ...")
            model = MultiDGDAdapter.train(adata, args.save_dir + os.sep, args.model_name,
                                          latent_dim=args.latent_dim,
                                          n_components=(n_ct if n_ct > 1 else None),
                                          n_epochs=args.n_epochs, random_seed=args.seed)
        _log("Learning test representations over all cells ...")
        model.learn_test_representations(adata, n_epochs=args.predict_epochs)
        _log(f"Computing deltas ({args.direction}, cell_batch={args.cell_batch}) ...")
        D, kept, p0, resp_names = perturbation.compute_deltas(
            model, groups, silenced_index, resp_index, args.direction,
            steps=args.perturb_steps, cell_batch=args.cell_batch, log=_log)
        np.savez(cache, D=D, kept=np.array(kept, dtype=object),
                 p0=p0, resp_names=np.array(resp_names, dtype=object))
        _log(f"Cached deltas -> {cache}")

    _log(f"Scoring {len(kept)} silenced features (bg={args.bg}) ...")
    records = scoring.score_from_deltas(D, kept, p0, groups, resp_index, args.direction,
                                        n_bg=args.n_bg, seed=args.seed,
                                        match_bg=(args.bg == "matched"), log=_log)
    df = scoring.records_to_p2g(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    _log(f"Saved: {args.out} ({len(df)} links)")


def prep(args):
    """Convert a .h5ad / .h5mu into the flat export directory that `link` reads."""
    import pandas as pd
    if args.mudata:
        adata = e2g_io.read_h5mu(args.mudata, rna=args.rna_mod, atac=args.atac_mod)
    else:
        adata = e2g_io.read_h5ad(args.adata)
    _log(f"{adata.n_obs} cells | {int((adata.var['modality'] == 'GEX').sum())} genes | "
         f"{int((adata.var['modality'] == 'ATAC').sum())} peaks")

    pairs = None
    if args.pairs:
        pairs = pd.read_csv(args.pairs)
    elif args.gtf:
        tss = e2g_io.tss_from_gtf(args.gtf)
        peaks_df = e2g_io.peaks_from_names(adata.var_names[(adata.var["modality"] == "ATAC").values])
        pairs = e2g_io.candidate_pairs_by_distance(tss, peaks_df, distance=args.distance)
        _log(f"generated {len(pairs)} candidate pairs (±{args.distance} bp of TSS)")

    e2g_io.write_export(adata, args.out, pairs=pairs,
                        observable=args.observable, covariate=args.covariate)
    note = "" if pairs is not None else "  (no pairs.csv — pass --pairs or --gtf, or add it before `link`)"
    _log(f"Wrote export -> {args.out}{note}")


def build_parser():
    p = argparse.ArgumentParser(prog="multidgd-e2g", description="Genome-wide enhancer-gene "
                                "linking via multiDGD in-silico perturbation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="convert .h5ad/.h5mu -> the flat export directory")
    src = pp.add_mutually_exclusive_group(required=True)
    src.add_argument("--adata", help="multi-modal .h5ad (var has modality/feature_types)")
    src.add_argument("--mudata", help=".h5mu with RNA + ATAC modalities")
    pp.add_argument("--out", required=True, help="output export directory")
    pp.add_argument("--observable", default="observable", help="obs column for cell type")
    pp.add_argument("--covariate", default=None, help="obs column for batch (-> covariate_sample)")
    pp.add_argument("--rna-mod", default="rna", help="MuData RNA modality key")
    pp.add_argument("--atac-mod", default="atac", help="MuData ATAC modality key")
    pgrp = pp.add_mutually_exclusive_group()
    pgrp.add_argument("--pairs", help="candidate pairs CSV (columns gene,peak)")
    pgrp.add_argument("--gtf", help="GTF to generate candidate pairs from gene TSS windows")
    pp.add_argument("--distance", type=int, default=250_000, help="TSS window for --gtf pairs")
    pp.set_defaults(func=prep)

    lk = sub.add_parser("link", help="run perturbation + scoring")
    lk.add_argument("--input-dir", required=True, help="flat export dir (see io.read_export)")
    lk.add_argument("--save-dir", required=True, help="dir for the model checkpoint + delta cache")
    lk.add_argument("--model-name", required=True)
    lk.add_argument("--out", required=True, help="output p2g .tsv")
    g = lk.add_mutually_exclusive_group()
    g.add_argument("--train", dest="load", action="store_false", help="train a new model")
    g.add_argument("--load", dest="load", action="store_true", help="load cached model (default)")
    lk.set_defaults(load=True)
    lk.add_argument("--direction", choices=["gene2peak", "peak2gene"], default="gene2peak")
    lk.add_argument("--bg", choices=["matched", "random"], default="matched",
                    help="null background. 'matched' (default): accessibility-matched, "
                         "controls for peak accessibility (like Signac/FigR). 'random': random "
                         "distal peaks -- more sensitive but its rankings are inflated by "
                         "promoter/high-accessibility peaks; use for benchmark parity only.")
    lk.add_argument("--batch-correct", action="store_true")
    lk.add_argument("--latent-dim", type=int, default=20)
    lk.add_argument("--n-epochs", type=int, default=500)
    lk.add_argument("--predict-epochs", type=int, default=20)
    lk.add_argument("--perturb-steps", type=int, default=1)
    lk.add_argument("--cell-batch", type=int, default=8192)
    lk.add_argument("--n-bg", type=int, default=500)
    lk.add_argument("--max-silenced", type=int, default=0)
    lk.add_argument("--train-fraction", type=float, default=0.8)
    lk.add_argument("--reuse-deltas", action="store_true")
    lk.add_argument("--seed", type=int, default=0)
    lk.set_defaults(func=link)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
