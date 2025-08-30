# Dynamic Focus Search & TSP Algorithms

This project implements, benchmarks, and visualizes advanced algorithms for nearest neighbor search and the Travelling Salesman Problem (TSP) in Python.

## Features

- **Dynamic Focus Search (DFS)**: Fast nearest neighbor search with temporal locality, compared to naive linear scan.
- **Greedy DFS TSP**: Heuristic TSP solver using DFS-based nearest neighbor.
- **DFLS 2-opt TSP**: Efficient TSP solver for large n (≥100) using a dynamic focus region and 2-opt swaps.
- **Christofides TSP**: Approximate TSP solver using NetworkX's Christofides algorithm.
- **Benchmarking**: Performance and solution quality comparison for all algorithms on random datasets of size n=50, 200, 500.
- **Visualization**: Side-by-side and grid visualizations of TSP tours for Greedy DFS, DFLS 2-opt, and Christofides algorithms.
- **Correctness Checks**: Ensures DFS matches naive scan for nearest neighbor queries.

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
