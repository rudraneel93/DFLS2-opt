

# Dynamic Focus Search & TSP Algorithms

This project implements, benchmarks, and visualizes advanced algorithms for nearest neighbor search and the Travelling Salesman Problem (TSP) in Python. It is designed for both research and practical use, with a focus on speed, scalability, and solution quality for large datasets.

## Features

- **Dynamic Focus Search (DFS)**: Fast nearest neighbor search with temporal locality, compared to naive linear scan. Includes correctness checks to ensure results match the naive baseline.
- **Greedy DFS TSP**: Heuristic TSP solver using DFS-based nearest neighbor, providing fast but suboptimal tours.
- **DFLS 2-opt TSP (Invented Algorithm)**: A novel, efficient TSP solver for large n (≥100) that combines a dynamic focus region with 2-opt swaps. The algorithm starts with a greedy DFS tour and iteratively improves it by performing 2-opt swaps, but only between cities that are close to each other (within a focus radius). This dramatically reduces unnecessary computations and accelerates convergence, making it practical for large-scale TSP instances.
- **Christofides TSP**: Approximate TSP solver using NetworkX's Christofides algorithm (run for n ≤ 1000 for speed).
- **Benchmarking**: Performance and solution quality comparison for all algorithms on random datasets of size n=50, 200, 500, 1000, 2000, 5000.
- **Visualization**: Side-by-side and grid visualizations of TSP tours for Greedy DFS, DFLS 2-opt, and Christofides algorithms. Includes scatter plots and bar charts for tour length and runtime comparisons.
- **Scalability**: DFLS 2-opt is optimized for large n (up to 5000+), with precomputed distances, cached tour costs, and efficient local updates.
- **Summary Table**: Prints a markdown table of all benchmark results for easy comparison and export.

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
- Visual comparison of TSP tours for each algorithm and city size (side-by-side and grid)
- Tour length vs. runtime scatter plot and bar charts
- Summary table of results (see below)

## Requirements
- Python 3.8+
- `matplotlib`, `networkx`

## Files
- `dfs.py`: Main implementation, benchmarking, and visualization
- `dsp_tsp.py`: (Optional) Placeholder for exact TSP solver
- `README.md`: This file

## Benchmark Results (n=50, 200, 500, 1000, 2000, 5000)

| n    | Greedy DFS Length | Greedy DFS Time (s) | DFLS 2-opt Length | DFLS 2-opt Time (s) | Christofides Length | Christofides Time (s) |
|------|-------------------|---------------------|-------------------|---------------------|---------------------|-----------------------|
| 50   | 12679.45          | 0.00                | 12246.96          | 0.01                | 10594.53            | 0.02                  |
| 200  | 25932.90          | 0.02                | 22449.34          | 0.07                | 22902.07            | 0.24                  |
| 500  | 39708.57          | 0.10                | 34697.02          | 0.85                | 36807.23            | 2.94                  |
| 1000 | 58673.80          | 0.29                | 49709.83          | 6.63                | 51804.67            | 27.29                 |
| 2000 | 81023.88          | 1.05                | 69568.10          | 33.47               | -                   | -                     |
| 5000 | 128324.34         | 6.47                | 116921.57         | 224.37              | -                   | -                     |

## Visualization Examples

- **Tour Plots**: Side-by-side and grid visualizations for each algorithm and city size.
- **Scatter Plot**: Tour length vs. runtime for all algorithms and sizes.
- **Bar Charts**: Tour length and runtime comparisons for all algorithms and sizes.

## License
MIT

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
