#!/usr/bin/env python3
"""
Main entry point: run all comparison experiments.

Usage:
    python run_all.py              # Run all experiments
    python run_all.py linear       # Linear manifold only
    python run_all.py nonlinear    # Circle manifold only
    python run_all.py image        # MNIST image experiment only
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode in ('all', 'linear'):
        print("\n" + "#" * 70)
        print("# EXPERIMENT 1: Linear Manifold (d=100, k=5)")
        print("#" * 70 + "\n")
        from experiments.compare_linear import run_experiment as run_linear
        run_linear(d=100, k=5)

    if mode in ('all', 'nonlinear'):
        print("\n" + "#" * 70)
        print("# EXPERIMENT 2: Nonlinear Circle Manifold (d=20, k=1)")
        print("#" * 70 + "\n")
        from experiments.compare_nonlinear import run_experiment as run_nonlinear
        run_nonlinear(d=20)

    if mode in ('all', 'image'):
        print("\n" + "#" * 70)
        print("# EXPERIMENT 3: MNIST Image (d=784)")
        print("#" * 70 + "\n")
        from experiments.compare_image import run_experiment as run_image
        run_image(digit=None)

    print("\n" + "#" * 70)
    print("# ALL EXPERIMENTS COMPLETE")
    print(f"# Figures saved to: {os.path.join(os.path.dirname(__file__), 'figures')}")
    print("#" * 70)


if __name__ == '__main__':
    main()
