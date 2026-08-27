#!/usr/bin/env Rscript
# seurat_to_export.R -- convert a Seurat multiome object into the flat export directory that
# `multidgd-e2g link` reads (rna_counts.mtx, atac_counts.mtx, rna_features.txt, atac_features.txt,
# barcodes.txt, obs.csv, and optionally pairs.csv).
#
# Usage:
#   Rscript seurat_to_export.R --rds obj.rds --out export_dir \
#       [--rna-assay RNA] [--atac-assay ATAC] \
#       [--celltype-col celltype] [--sample-col sample] \
#       [--pairs pairs.csv | --genebed genes.bed --distance 250000]
#
# --celltype-col -> obs 'observable' (sets the number of GMM components).
# --sample-col   -> obs 'covariate_sample' (batch correction; optional).
# Candidate pairs: pass --pairs (columns gene,peak) OR --genebed (BED: chr,start,end,gene[,.,strand])
# to generate peaks within +/- --distance of each gene's TSS (needs GenomicRanges).

suppressPackageStartupMessages({ library(Matrix); library(SeuratObject) })

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag); if (length(i)) args[i + 1] else default
}
rds        <- getarg("--rds")
out_dir    <- getarg("--out")
rna_assay  <- getarg("--rna-assay", "RNA")
atac_assay <- getarg("--atac-assay", "ATAC")
ct_col     <- getarg("--celltype-col")
samp_col   <- getarg("--sample-col")
pairs_csv  <- getarg("--pairs")
genebed    <- getarg("--genebed")
distance   <- as.integer(getarg("--distance", "250000"))
if (is.null(rds) || is.null(out_dir)) stop("--rds and --out are required")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obj <- readRDS(rds)

rna  <- SeuratObject::GetAssayData(obj, assay = rna_assay,  layer = "counts")   # genes x cells
atac <- SeuratObject::GetAssayData(obj, assay = atac_assay, layer = "counts")   # peaks x cells
stopifnot(identical(colnames(rna), colnames(atac)))

invisible(Matrix::writeMM(as(rna,  "CsparseMatrix"), file.path(out_dir, "rna_counts.mtx")))
invisible(Matrix::writeMM(as(atac, "CsparseMatrix"), file.path(out_dir, "atac_counts.mtx")))
writeLines(rownames(rna),  file.path(out_dir, "rna_features.txt"))
writeLines(rownames(atac), file.path(out_dir, "atac_features.txt"))
writeLines(colnames(rna),  file.path(out_dir, "barcodes.txt"))

obs <- data.frame(barcode = colnames(rna), stringsAsFactors = FALSE)
obs$observable <- if (!is.null(ct_col) && ct_col %in% colnames(obj@meta.data))
  as.character(obj@meta.data[[ct_col]]) else "cells"
if (!is.null(samp_col) && samp_col %in% colnames(obj@meta.data))
  obs$covariate_sample <- as.character(obj@meta.data[[samp_col]])
write.csv(obs, file.path(out_dir, "obs.csv"), row.names = FALSE)

# --- candidate pairs ---
if (!is.null(pairs_csv)) {
  file.copy(pairs_csv, file.path(out_dir, "pairs.csv"), overwrite = TRUE)
  cat("copied pairs ->", file.path(out_dir, "pairs.csv"), "\n")
} else if (!is.null(genebed)) {
  suppressPackageStartupMessages(library(GenomicRanges))
  bed <- read.table(genebed, header = FALSE, stringsAsFactors = FALSE)
  gchr <- bed[[1]]; gstart <- as.integer(bed[[2]]); gend <- as.integer(bed[[3]]); gname <- bed[[4]]
  strand <- if (ncol(bed) >= 6) bed[[6]] else rep("+", nrow(bed))
  tss <- ifelse(strand == "-", gend, gstart)
  gr_gene <- GRanges(gchr, IRanges(pmax(tss - distance, 1), tss + distance), gene = gname)
  pk <- do.call(rbind, strsplit(rownames(atac), "-"))
  # peak represented by its center, matching io.candidate_pairs_by_distance (the Python path)
  pk_center <- (as.integer(pk[, 2]) + as.integer(pk[, 3])) %/% 2L
  gr_peak <- GRanges(pk[, 1], IRanges(pk_center, pk_center), peak = rownames(atac))
  ov <- findOverlaps(gr_gene, gr_peak)
  pairs <- data.frame(gene = gr_gene$gene[queryHits(ov)],
                      peak = gr_peak$peak[subjectHits(ov)], stringsAsFactors = FALSE)
  write.csv(pairs, file.path(out_dir, "pairs.csv"), row.names = FALSE)
  cat("generated", nrow(pairs), "pairs ->", file.path(out_dir, "pairs.csv"), "\n")
} else {
  cat("NOTE: no --pairs or --genebed; write pairs.csv (gene,peak) before running `link`.\n")
}

cat(sprintf("Wrote export -> %s (%d genes x %d cells, %d peaks)\n",
            out_dir, nrow(rna), ncol(rna), nrow(atac)))
