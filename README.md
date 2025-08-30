
# Dynamic Focus Search (DFS) & Dynamic Focused Local Search (DFLS) Python Project

This project implements and benchmarks:
- The Dynamic Focus Search (DFS) algorithm for nearest neighbor search
- A greedy DFS-based TSP solver
- The novel Dynamic Focused Local Search (DFLS) 2-opt algorithm for large-scale TSP

## DFLS 2-opt: Motivation & Description

Solving the Travelling Salesman Problem (TSP) for large numbers of cities is computationally challenging. Classic exact solvers are only practical for small n, while greedy heuristics are fast but often suboptimal. The **DFLS 2-opt** algorithm is designed to efficiently improve TSP tours for large n (≥100) by combining:

1. **Greedy DFS Initialization:** Quickly generates an initial tour by always visiting the nearest unvisited city.
2. **Dynamic Focused Local Search:** Iteratively applies 2-opt swaps, but restricts swaps to city pairs within a specified "focus region" (distance threshold). This dramatically reduces computation while targeting the most promising improvements.

### How DFLS 2-opt Works
1. Start with a greedy DFS tour.
2. For each pair of cities (i, j) within the tour:
    - If the distance between city i and city j is less than `focus_radius`, consider a 2-opt swap (reverse the segment between i+1 and j).
    - Accept the swap if it reduces the total tour length.
3. Repeat until no further improvements are found or a maximum number of iterations is reached.

### Why DFLS 2-opt is Effective
- **Scalability:** Focus region limits the number of swaps, making the algorithm fast for large n.
- **Quality:** Produces tours significantly shorter than greedy DFS, as shown in benchmarks and visualizations.
- **Novelty:** The dynamic focus region is a practical innovation for balancing speed and solution quality.

## TSP Algorithm Comparison

| Algorithm         | Tour Length (n=100) | Runtime (seconds) |
|-------------------|--------------------|-------------------|
| Greedy DFS        | 19843.78           | Fast              |
| DFLS 2-opt        | 18096.97           | Fast              |

*Note: Both algorithms use the same set of 100 cities for fair comparison. DFLS 2-opt consistently produces shorter tours.*

## How to Run

1. Open the workspace in VS Code.
2. Ensure Python and required extensions are installed.
3. Run the main script to benchmark DFS, greedy DFS TSP, and DFLS 2-opt TSP.

## Files
- `dfs.py`: Main implementation and benchmarking script
- `dsp_tsp.py`: Exact TSP solver for small n
- `.github/copilot-instructions.md`: Copilot custom instructions

---

Feel free to modify or extend the project for your own experiments.
