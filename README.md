
# Dynamic Focus Search & TSP Algorithms

This project implements, benchmarks, and visualizes advanced algorithms for nearest neighbor search and the Travelling Salesman Problem (TSP) in Python.

## Features

- **Dynamic Focus Search (DFS)**: Fast nearest neighbor search with temporal locality, compared to naive linear scan.
- **Greedy DFS TSP**: Heuristic TSP solver using DFS-based nearest neighbor.
- **DFLS 2-opt TSP (Invented Algorithm)**: A novel, efficient TSP solver for large n (≥100) that combines a dynamic focus region with 2-opt swaps. The algorithm starts with a greedy DFS tour and iteratively improves it by performing 2-opt swaps, but only between cities that are close to each other (within a focus radius). This dramatically reduces unnecessary computations and accelerates convergence, making it practical for large-scale TSP instances.
- **Christofides TSP**: Approximate TSP solver using NetworkX's Christofides algorithm.
- **Benchmarking**: Performance and solution quality comparison for all algorithms on random datasets of size n=50, 200, 500.
- **Visualization**: Side-by-side and grid visualizations of TSP tours for Greedy DFS, DFLS 2-opt, and Christofides algorithms.
- **Correctness Checks**: Ensures DFS matches naive scan for nearest neighbor queries.

## Invented Algorithm: DFLS 2-opt TSP

**DFLS 2-opt TSP** is a new algorithm designed for efficient TSP solving on large datasets. It leverages a dynamic focus region to restrict 2-opt swaps to city pairs that are spatially close, greatly reducing the search space and computation time. This approach combines the speed of greedy heuristics with the solution quality improvements of local search, making it highly effective for n ≥ 100.

**How it works:**
- Start with a greedy DFS tour.
- Iteratively perform 2-opt swaps, but only between cities within a specified focus radius.
- Continue until no further improvements are found or a maximum number of iterations is reached.
- Returns a high-quality tour much faster than standard 2-opt for large n.

## Usage

Run the main script:

```bash
python3 dfs.py
```

Or, if using the provided virtual environment:

```bash
./.venv-1/bin/python dfs.py
```

## Output
- Benchmark results for nearest neighbor search (DFS vs. Naive)
- TSP tour lengths and runtimes for Greedy DFS, DFLS 2-opt, and Christofides
- Visual comparison of TSP tours for each algorithm and city size
- Summary table of results

## Requirements
- Python 3.8+
- `matplotlib`, `networkx`

## Files
- `dfs.py`: Main implementation, benchmarking, and visualization
- `dsp_tsp.py`: (Optional) Placeholder for exact TSP solver
- `README.md`: This file

## License
MIT

## Usage

Run the main script:

```bash
python3 dfs.py
```

Or, if using the provided virtual environment:

```bash
./.venv-1/bin/python dfs.py
```

## Output
- Benchmark results for nearest neighbor search (DFS vs. Naive)
- TSP tour lengths and runtimes for Greedy DFS, DFLS 2-opt, and Christofides
- Visual comparison of TSP tours for each algorithm and city size
- Summary table of results

## Requirements
- Python 3.8+
- `matplotlib`, `networkx`

## Files
- `dfs.py`: Main implementation, benchmarking, and visualization
- `dsp_tsp.py`: (Optional) Placeholder for exact TSP solver
- `README.md`: This file

## License
MIT
