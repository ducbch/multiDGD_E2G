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
        self.model = model

    # --- construction / training (multiDGD's own code) ---
    @classmethod
    def load(cls, adata, save_dir, model_name):
        import multiDGD
        return cls(multiDGD.DGD.load(data=adata, save_dir=save_dir, model_name=model_name))

    @classmethod
    def train(cls, adata, save_dir, model_name, latent_dim=20, n_components=None,
              n_epochs=500, random_seed=0):
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
        return self.model.test_set.library

    @property
    def modality_switch(self) -> int:
        return int(self.model.test_set.modality_switch)

    @property
    def test_rep(self):
        return self.model.test_rep

    @property
    def has_correction(self) -> bool:
        return (getattr(self.model, "correction_gmm", None) is not None) and \
               (getattr(self.model, "correction_test_rep", None) is not None)

    @property
    def correction_z(self):
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
