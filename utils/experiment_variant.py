"""Helpers for keeping rerun artifacts separate from published results."""

import os


def output_dir(source_file, kind):
    """Return the output directory, optionally namespaced by EXPERIMENT_VARIANT."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(source_file)))
    path = os.path.join(root, kind)
    variant = os.environ.get("EXPERIMENT_VARIANT", "").strip()
    if variant:
        path = os.path.join(path, variant)
    os.makedirs(path, exist_ok=True)
    return path


def experiment_variant():
    """Human-readable variant name used in logs and result metadata."""
    return os.environ.get("EXPERIMENT_VARIANT", "").strip() or "default"


def paired_stage2_t_min(t0):
    """Stage-2 lower bound for the active controlled-comparison variant."""
    if experiment_variant() == "sild_matches_ldm":
        return 0.001
    return float(t0)
