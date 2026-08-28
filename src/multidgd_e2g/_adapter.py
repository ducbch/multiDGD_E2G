"""Adapter isolating every multiDGD-internal touchpoint behind a stable interface.

multiDGD_E2G consumes a trained multiDGD model but calls some of its internals (the
decoder, the learned representations, the test dataset tensors). Keeping all of that in
this one module means an upstream multiDGD API change only requires fixing this file.

The scientific model (decoder, GMM, representations, training) is entirely multiDGD's --
this adapter only wires it to the perturbation engine.
"""
from __future__ import annotations

import torch


class MultiDGDAdapter:
    """Thin wrapper over a trained ``multiDGD.DGD`` model."""

    def __init__(self, model):
        """Wrap an already-constructed ``multiDGD.DGD`` instance.

        Parameters
        ----------
        model : multiDGD.DGD
            A trained (or freshly built) multiDGD model.
        """
        self.model = model

    # --- construction / training (multiDGD's own code) ---
    @classmethod
    def load(cls, adata, save_dir, model_name):
        """Load a trained multiDGD checkpoint and wrap it.

        Parameters
        ----------
        adata : anndata.AnnData
            The dataset the model was trained on (same object layout as training).
        save_dir : str
            Directory holding ``<model_name>.pt`` and ``<model_name>_hyperparameters.json``.
        model_name : str
            Base name of the saved model.

        Returns
        -------
        MultiDGDAdapter
            Adapter wrapping the loaded model.
        """
        import multiDGD
        return cls(multiDGD.DGD.load(data=adata, save_dir=save_dir, model_name=model_name))

    @classmethod
    def train(cls, adata, save_dir, model_name, latent_dim=20, n_components=None,
              n_epochs=500, random_seed=0):
        """Train a new multiDGD model, save it, and wrap it.

        Parameters
        ----------
        adata : anndata.AnnData
            Multi-modal [RNA|ATAC] dataset (see :func:`io.build_anndata`).
        save_dir : str
            Directory to write the checkpoint into.
        model_name : str
            Base name for the saved model.
        latent_dim : int, optional
            Latent dimensionality (default 20).
        n_components : int or None, optional
            Number of GMM components; pass the cell-type count, or None to let multiDGD choose.
        n_epochs : int, optional
            Training epoch cap; early stopping may finish sooner (default 500).
        random_seed : int, optional
            Seed for reproducibility (default 0).

        Returns
        -------
        MultiDGDAdapter
            Adapter wrapping the trained, saved model.
        """
        import multiDGD
        params = {"latent_dimension": int(latent_dim)}
        if n_components is not None:
            params["n_components"] = int(n_components)
        model = multiDGD.DGD(data=adata, parameter_dictionary=params,
                             save_dir=save_dir, model_name=model_name, random_seed=random_seed)
        model.train(n_epochs=n_epochs)
        model.save()
        return cls(model)

    def learn_test_representations(self, adata, n_epochs=20):
        """Learn per-cell latent representations for the given cells (multiDGD's predict)."""
        self.model.predict(testdata=adata, n_epochs=n_epochs)

    # --- accessors used by the perturbation engine ---
    @property
    def counts(self):
        """Dense n_cells x total_features count tensor on CPU (kept off the GPU)."""
        return self.model.test_set.data

    @property
    def library(self):
        """Per-cell library sizes (n_cells x n_modalities) used to scale the decoder output."""
        return self.model.test_set.library

    @property
    def modality_switch(self) -> int:
        """Column index separating the RNA block from the ATAC block in the count matrix."""
        return int(self.model.test_set.modality_switch)

    @property
    def test_rep(self):
        """The learned per-cell latent representation (multiDGD ``RepresentationLayer``)."""
        return self.model.test_rep

    @property
    def has_correction(self) -> bool:
        """Whether the model carries a covariate (batch) correction model and its test rep."""
        return (getattr(self.model, "correction_gmm", None) is not None) and \
               (getattr(self.model, "correction_test_rep", None) is not None)

    @property
    def correction_z(self):
        """Detached correction (covariate) latent, or None if the model has no correction."""
        return self.model.correction_test_rep.z.detach() if self.has_correction else None

    def decode_z(self, z_rows, corr_rows):
        """Decode a subset of cells from their latent rows (+ correction rows).

        Mirrors ``predict_from_representation`` on a slice. Returns a list [gex, atac] of
        per-cell softmax outputs.
        """
        z = z_rows if corr_rows is None else torch.cat((z_rows, corr_rows), dim=1)
        return self.model.decoder(z)

    def decoder_loss(self, y_mod, target, scale, mod_id, gene_id):
        """multiDGD's reconstruction loss for one feature (used by the rep refit)."""
        return self.model.decoder.loss(y_mod, target, scale=scale, mod_id=mod_id, gene_id=gene_id)
