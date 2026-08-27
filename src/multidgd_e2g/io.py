"""I/O: build the AnnData multiDGD expects, convert between formats, generate candidate
peak-gene pairs, and group them for a perturbation direction.

Ingestion paths:
  * `read_export` reads the flat mtx/csv export directory `link` consumes
    (rna_counts.mtx, atac_counts.mtx, features, barcodes, obs, [pairs]).
  * `read_h5ad` / `read_h5mu` load an AnnData / MuData object.
  * `build_anndata` assembles the AnnData from in-memory matrices (for a pure-Python caller).
  * `write_export` writes the flat export directory from an AnnData (used by `prep`; the
    Seurat path lives in scripts/seurat_to_export.R).

Candidate pairs are peaks within +/-`distance` of a gene's TSS. Supply your own pairs table
(gene, peak) or generate one with `candidate_pairs_by_distance` (`tss_from_gtf` +
`peaks_from_names` extract the coordinates it needs).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def build_anndata(rna_counts, atac_counts, genes, peaks, barcodes, obs,
                  observable=None, covariate=None):
    """Assemble the [RNA|ATAC] AnnData multiDGD expects.

    rna_counts/atac_counts: (features x cells) sparse; genes/peaks/barcodes: name lists;
    obs: DataFrame indexed like barcodes. `observable` (cell-type col) drives GMM components;
    `covariate` (sample/batch col) becomes a multiDGD correction factor (prefixed covariate_).
    """
    import anndata as ad
    X = sp.hstack([rna_counts.T.tocsr(), atac_counts.T.tocsr()]).tocsr()
    var = pd.DataFrame(index=list(genes) + list(peaks))
    var["modality"] = ["GEX"] * len(genes) + ["ATAC"] * len(peaks)
    var["feature_types"] = var["modality"]
    obs = obs.copy()
    obs.index = list(barcodes)
    if observable is not None and observable in obs.columns:
        obs["observable"] = obs[observable].astype(str)
    elif "observable" not in obs.columns:
        obs["observable"] = "cells"
    if covariate is not None and covariate in obs.columns:
        obs["covariate_sample"] = obs[covariate].astype(str)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def read_export(in_dir):
    """Read the flat export directory into (adata, genes, peaks, pairs_or_None)."""
    import anndata as ad
    rna = scipy.io.mmread(os.path.join(in_dir, "rna_counts.mtx")).tocsr()    # genes x cells
    atac = scipy.io.mmread(os.path.join(in_dir, "atac_counts.mtx")).tocsr()  # peaks x cells
    genes = pd.read_csv(os.path.join(in_dir, "rna_features.txt"), header=None)[0].astype(str).tolist()
    peaks = pd.read_csv(os.path.join(in_dir, "atac_features.txt"), header=None)[0].astype(str).tolist()
    barcodes = pd.read_csv(os.path.join(in_dir, "barcodes.txt"), header=None)[0].astype(str).tolist()
    obs = pd.read_csv(os.path.join(in_dir, "obs.csv"))
    X = sp.hstack([rna.T, atac.T]).tocsr()
    var = pd.DataFrame(index=genes + peaks)
    var["modality"] = ["GEX"] * len(genes) + ["ATAC"] * len(peaks)
    var["feature_types"] = var["modality"]
    obs.index = barcodes
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    pairs_path = os.path.join(in_dir, "pairs.csv")
    pairs = pd.read_csv(pairs_path) if os.path.exists(pairs_path) else None
    return adata, genes, peaks, pairs


def assign_train_val_split(adata, train_fraction=0.8, seed=0, observable="observable"):
    """Stratified train/validation split (no test label, so predict scores ALL cells).

    multiDGD requires a validation set (and it errors on the correction model without one).
    """
    from sklearn.model_selection import train_test_split
    n = adata.n_obs
    strat = None
    if observable in adata.obs.columns:
        vc = adata.obs[observable].astype(str).value_counts()
        if vc.min() >= 2 and vc.size > 1:
            strat = adata.obs[observable].astype(str).values
    _, va = train_test_split(np.arange(n), test_size=1.0 - train_fraction,
                             random_state=seed, stratify=strat)
    lab = np.array(["train"] * n, dtype=object)
    lab[va] = "validation"
    adata.obs["train_val_test"] = lab
    return adata


def candidate_pairs_by_distance(gene_tss, peaks, distance=250_000):
    """Peaks within +/-`distance` of a gene TSS -> candidate (gene, peak) pairs.

    gene_tss : DataFrame with columns [gene, chr, tss].
    peaks    : DataFrame with columns [peak, chr, start, end] (peak id like 'chr1-100-200').
    Returns a DataFrame [gene, peak]. Efficient per-chromosome via sorted searchsorted.
    """
    pk = peaks.copy()
    pk["center"] = (pk["start"].astype(int) + pk["end"].astype(int)) // 2
    out_g, out_p = [], []
    for chrom, gsub in gene_tss.groupby("chr"):
        psub = pk[pk["chr"] == chrom].sort_values("center")
        if psub.empty:
            continue
        centers = psub["center"].to_numpy()
        pids = psub["peak"].to_numpy()
        for gene, tss in zip(gsub["gene"].to_numpy(), gsub["tss"].to_numpy()):
            lo = np.searchsorted(centers, tss - distance, side="left")
            hi = np.searchsorted(centers, tss + distance, side="right")
            if hi > lo:
                out_p.extend(pids[lo:hi].tolist())
                out_g.extend([gene] * (hi - lo))
    return pd.DataFrame({"gene": out_g, "peak": out_p})


_GEX = {"GEX", "gex", "RNA", "rna", "Gene Expression", "gene expression", "expression"}
_ATAC = {"ATAC", "atac", "Peaks", "peaks", "accessibility"}


def _normalize_modality(adata):
    """Ensure adata.var has a 'modality' column of 'GEX'/'ATAC' (from modality/feature_types)."""
    col = "modality" if "modality" in adata.var else ("feature_types" if "feature_types" in adata.var else None)
    if col is None:
        raise ValueError("AnnData var needs a 'modality' or 'feature_types' column (GEX/ATAC).")
    m = adata.var[col].astype(str).map(lambda v: "GEX" if v in _GEX else ("ATAC" if v in _ATAC else v))
    if not set(m.unique()) <= {"GEX", "ATAC"}:
        raise ValueError(f"Unrecognized modality labels: {sorted(set(m.unique()) - {'GEX','ATAC'})}")
    adata.var["modality"] = m.values
    return adata


def read_h5ad(path):
    """Read a single multi-modal AnnData (.h5ad); var must carry modality/feature_types."""
    import anndata as ad
    return _normalize_modality(ad.read_h5ad(path))


def read_h5mu(path, rna="rna", atac="atac"):
    """Read a MuData (.h5mu) with RNA + ATAC modalities into one AnnData with var['modality']."""
    import anndata as ad
    import mudata
    md = mudata.read(path)
    rna_ad, atac_ad = md.mod[rna], md.mod[atac]
    X = sp.hstack([rna_ad.X.tocsr(), atac_ad.X.tocsr()]).tocsr()
    var = pd.DataFrame(index=list(rna_ad.var_names) + list(atac_ad.var_names))
    var["modality"] = ["GEX"] * rna_ad.n_vars + ["ATAC"] * atac_ad.n_vars
    var["feature_types"] = var["modality"]
    adata = ad.AnnData(X=X, obs=md.obs.copy(), var=var)
    adata.var_names_make_unique()
    return adata


def write_export(adata, out_dir, pairs=None, observable="observable", covariate=None):
    """Write the flat export directory `link` reads, from a multi-modal AnnData.

    Splits var['modality'] into RNA/ATAC (raw-count X expected), and writes counts (MatrixMarket),
    feature/barcode lists, obs.csv (barcode + observable + optional covariate_sample), and pairs.csv.
    """
    adata = _normalize_modality(adata)
    os.makedirs(out_dir, exist_ok=True)
    gmask = (adata.var["modality"] == "GEX").values
    pmask = (adata.var["modality"] == "ATAC").values
    X = adata.X.tocsr()
    scipy.io.mmwrite(os.path.join(out_dir, "rna_counts.mtx"), X[:, gmask].T.tocoo())    # genes x cells
    scipy.io.mmwrite(os.path.join(out_dir, "atac_counts.mtx"), X[:, pmask].T.tocoo())   # peaks x cells
    genes = adata.var_names[gmask].tolist()
    peaks = adata.var_names[pmask].tolist()
    pd.Series(genes).to_csv(os.path.join(out_dir, "rna_features.txt"), index=False, header=False)
    pd.Series(peaks).to_csv(os.path.join(out_dir, "atac_features.txt"), index=False, header=False)
    pd.Series(adata.obs_names).to_csv(os.path.join(out_dir, "barcodes.txt"), index=False, header=False)
    obs = pd.DataFrame({"barcode": list(adata.obs_names)})
    obs["observable"] = adata.obs[observable].astype(str).values if observable in adata.obs else "cells"
    if covariate is not None and covariate in adata.obs:
        obs["covariate_sample"] = adata.obs[covariate].astype(str).values
    obs.to_csv(os.path.join(out_dir, "obs.csv"), index=False)
    if pairs is not None:
        pairs[["gene", "peak"]].to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    return out_dir


def peaks_from_names(peak_names):
    """Parse 'chr-start-end' peak ids into a DataFrame [peak, chr, start, end]."""
    rows = []
    for p in peak_names:
        chrom, start, end = p.rsplit("-", 2)
        rows.append((p, chrom, int(start), int(end)))
    return pd.DataFrame(rows, columns=["peak", "chr", "start", "end"])


def tss_from_gtf(gtf_path):
    """Extract gene TSS from a GTF -> DataFrame [gene, chr, tss] (TSS = start on +, end on -)."""
    import re
    name_re = re.compile(r'gene_name "([^"]+)"')
    seen, rows = set(), []
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = name_re.search(f[8])
            if not m:
                continue
            gene, chrom, strand = m.group(1), f[0], f[6]
            if gene in seen:
                continue
            seen.add(gene)
            tss = int(f[3]) if strand == "+" else int(f[4])
            rows.append((gene, chrom, tss))
    return pd.DataFrame(rows, columns=["gene", "chr", "tss"])


def build_groups(pairs, gene_row, peak_col, direction, max_silenced=0):
    """Group candidate pairs by the silenced feature for a direction; return (groups, silenced_index, resp_index)."""
    pairs = pairs[pairs["gene"].isin(gene_row) & pairs["peak"].isin(peak_col)].copy()
    if direction == "gene2peak":
        groups = pairs.groupby("gene")["peak"].apply(list).to_dict()
        silenced_index, resp_index = gene_row, peak_col
    else:  # peak2gene
        groups = pairs.groupby("peak")["gene"].apply(list).to_dict()
        silenced_index, resp_index = peak_col, gene_row
    if max_silenced and max_silenced < len(groups):
        keep = list(groups.keys())[:max_silenced]
        groups = {k: groups[k] for k in keep}
    return groups, silenced_index, resp_index
