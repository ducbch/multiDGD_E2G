"""multiDGD_E2G: genome-wide enhancer-gene linking via multiDGD in-silico perturbation.

multiDGD (Schuster et al., Nat Commun 2024) introduced in-silico perturbation as a proof of
concept for linking regulatory regions to genes. multiDGD_E2G develops that into a scalable,
genome-wide enhancer-gene linking tool: it silences each feature on a trained multiDGD model,
reads the partner modality's predicted change (mini-batched over cells so memory scales with the batch size,
not the cell count), and scores the change into calibrated link statistics.

Public API:
    from multidgd_e2g import MultiDGDAdapter, compute_deltas, score_from_deltas, records_to_p2g
    from multidgd_e2g import io
"""
from .scoring import score_from_deltas, records_to_p2g, bh_fdr
from . import io

# torch-backed symbols are imported lazily so I/O (e.g. `prep`) works without the GPU stack.
_LAZY = {"MultiDGDAdapter": "._adapter",
         "compute_deltas": ".perturbation", "compute_cell_deltas": ".perturbation"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MultiDGDAdapter", "compute_deltas", "compute_cell_deltas",
    "score_from_deltas", "records_to_p2g", "bh_fdr", "io",
]
__version__ = "0.1.0"
