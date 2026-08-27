# Examples (input conversion + smoke test)

A small, self-contained CD14 monocyte multiome subset from two PBMC samples (5,486 cells,
chr1q21 region around the S100A8/A9 marker cluster: 250 genes × 998 peaks, 16,053 candidate
pairs), in four formats, for testing the input converters and a quick CLI smoke test.

For the real, genome-wide analysis, see [`../workflow/`](../workflow/WORKFLOW_README.md).

## Contents

```
examples/
  standard_input/    # the flat export directory the CLI reads (the target format)
  anndata_input/     # cd14_example.h5ad   (AnnData; var.feature_types = GEX/ATAC)
  mudata_input/      # cd14_example.h5mu   (MuData; rna + atac modalities)
  seurat_input/      # cd14_example.rds    (Seurat; RNA + ATAC assays)
  genes.bed  genes.gtf   # gene models for candidate-pair generation
```

The three `*_input/` objects and `standard_input/` hold the **same** subset, so each converter
reproduces `standard_input/` byte-for-byte (same counts, features, barcodes, and 16,053 pairs).
Each cell carries its sample label (`covariate_sample`) for batch correction.

## 1. Convert any input format to the standard export

```bash
# AnnData
multidgd-e2g prep --adata anndata_input/cd14_example.h5ad --out /tmp/exp_adata \
  --observable celltype --covariate sample --gtf genes.gtf

# MuData
multidgd-e2g prep --mudata mudata_input/cd14_example.h5mu --out /tmp/exp_mudata \
  --observable celltype --covariate sample --gtf genes.gtf

# Seurat
Rscript ../scripts/seurat_to_export.R --rds seurat_input/cd14_example.rds --out /tmp/exp_seurat \
  --celltype-col celltype --sample-col sample --genebed genes.bed
```

Each `/tmp/exp_*` directory matches `standard_input/`.

## 2. Quick CLI smoke test (optional, minutes on CPU)

Check the install end to end on the small `standard_input/`. This is a **mechanics test only** —
the region is tiny, so the scores are not meaningful (model, deltas, and null are all confined to
this locus). For real links use the genome-wide workflow.

```bash
multidgd-e2g link --train --input-dir standard_input --save-dir /tmp/smoke \
  --model-name dgd_smoke --batch-correct --n-epochs 100 --cell-batch 2048 \
  --out /tmp/smoke_gene2peak.tsv
```

## Notes

- Candidate pairs use the peak **center** within ±250 kb of a gene TSS (consistent across all
  three converters).
- This subset is for conversion/smoke-testing only; it is not a valid analysis. See
  [`../workflow/`](../workflow/WORKFLOW_README.md) for the genome-wide run.
