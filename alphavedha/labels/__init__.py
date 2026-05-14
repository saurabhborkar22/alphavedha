"""Labels — triple barrier labeling and sample weighting."""

from alphavedha.labels.sample_weights import compute_sample_weights
from alphavedha.labels.triple_barrier import LabelResult, compute_triple_barrier_labels

__all__ = ["LabelResult", "compute_sample_weights", "compute_triple_barrier_labels"]
