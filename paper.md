---
title: 'Dynamic Focus Search & Scalable TSP Algorithms: Efficient Large-Scale Optimization in Python'
authors:
  - name: Rudraneel Das
    orcid: 0009-0009-6173-0262
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2025-09-01
keywords:
  - algorithms
  - optimization
  - travelling salesman problem
  - data science
  - python
  - combinatorial optimization
subject: Algorithms
---

# Summary

Dynamic Focus Search & Scalable TSP Algorithms is a Python package for efficient nearest neighbor search and solving the Travelling Salesman Problem (TSP) at scale. It introduces the DFLS 2-opt algorithm, which leverages a dynamic focus region to accelerate local search, making it practical for datasets with thousands of cities. The package includes benchmarking, visualization, and comparison with Greedy DFS and Christofides algorithms.

# Statement of need

Solving large-scale TSP instances is critical in logistics, robotics, and data science. Existing heuristics struggle with scalability and speed. This package provides a novel, scalable solution with high-quality results and advanced visualization tools, enabling rapid experimentation and deployment.

# Functionality

- Dynamic Focus Search (DFS) for fast nearest neighbor search.
- Greedy DFS and DFLS 2-opt TSP solvers.
- Christofides TSP via NetworkX.
- Benchmarking and visualization (side-by-side, grid, scatter, bar charts).
- Optimized for large n (up to 5000+ cities).

# Example usage

```bash
python3 dfs.py
```

# References

- Lawler, E. L., et al. "The Traveling Salesman Problem: A Guided Tour of Combinatorial Optimization." Wiley, 1985.
- Christofides, N. "Worst-case analysis of a new heuristic for the travelling salesman problem." 1976.
- NetworkX Documentation: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman_problem.christofides.html

# Acknowledgements

Thanks to the open-source community and contributors.
