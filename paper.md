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
subject: Discrete Mathematics and Combinatorics
---

# Summary

Dynamic Focus Search & Scalable TSP Algorithms is a Python package for efficient nearest neighbor search and scalable solutions to the Travelling Salesman Problem (TSP). It introduces the DFLS 2-opt algorithm, which leverages a dynamic focus region to accelerate local search, making it practical for datasets with thousands of cities. The package includes benchmarking, visualization, and comparison with Greedy DFS and Christofides algorithms.

# Introduction

Combinatorial optimization and graph traversal tasks underpin many practical systems: route planning, task scheduling, resource allocation, and constraint satisfaction. Although exhaustive methods such as depth-first search (DFS) guarantee completeness, they frequently suffer from combinatorial explosion in realistic problem instances. Heuristic-driven metaheuristics (simulated annealing, tabu search, genetic algorithms) tend to locate high-quality solutions but can be computationally expensive and difficult to reproduce. DFLS2-opt proposes a middle path: preserve the structural guarantees of DFS while injecting lightweight local heuristics and adaptive pruning rules to reduce wasteful exploration. The resulting framework is simple to implement, interpretable, and extensible for applied use and research experimentation.

# Related Work

Depth-first strategies and local search techniques are classical approaches in algorithm design. DFS provides a controlled traversal order (see Knuth, 1997), while local search and stochastic techniques improve empirical performance on large instances (Kirkpatrick et al., 1983; Hoos & Stützle, 2004). Hybridizations that combine systematic traversal with heuristic guidance have been explored in constraint programming and memetic algorithms. DFLS2-opt builds on these foundations by focusing on simple, reproducible mechanisms suitable for open-source release and community validation.

# Methodology

DFLS2-opt is organized around four core components:
- **Depth-First Exploration:** The algorithm performs controlled, stack-based traversal of the search tree, ensuring completeness under specified bounds.
- **Heuristic Local Adjustments:** At branching points, candidate moves are scored using lightweight heuristics (e.g., greedy delta, degree-based scores, or problem-specific cost estimates) to prioritize promising directions.
- **Pruning & Backtracking Optimization:** Adaptive thresholds are applied to prune subtrees whose partial-cost bounds exceed dynamically adjusted cutoffs. These thresholds are tightened or relaxed based on observed success rates.
- **Feedback-Driven Adaptation:** Runtime statistics (best-so-far, frequency of improvements, depth of successful probes) inform parameter updates for future branches, making the search progressively focused on fruitful regions of the space.

Implementation details:
- The core is implemented in modular Python; key modules include search control, heuristic scorers, pruning policy, and data loaders for benchmarks.
- Interfaces are provided to plug in problem-specific evaluators (e.g., TSP distance functions, scheduling penalty calculators).
- A compact experimental harness automates dataset loading, repeated trials, statistical aggregation, and result plotting.

**Algorithm (high level):**
1. Initialize stack with initial state and parameters.
2. While stack not empty and budget remains:
   a. Pop state S from stack.
   b. If S is a complete solution, evaluate and update best-so-far.
   c. Generate candidate successor moves from S.
   d. Score candidates using heuristic scorers and sort by priority.
   e. For each candidate C:
      i. If pruning_condition(C) holds, skip.
      ii. Push C onto stack (with adjusted parameters).
3. Periodically update pruning thresholds and heuristics based on observed outcomes.

# Results & Evaluation

We evaluated DFLS2-opt on three categories of benchmarks to demonstrate generality:
- **Synthetic random graphs** (Erdős–Rényi and Barabási–Albert families) scaled from 100 to 10,000 nodes.
- **Real-world route networks** derived from open transportation datasets (small city road networks).
- **Scheduling benchmarks** adapted from standard CSP instances.

Experimental setup:
- Each instance was run 30 times with different random seeds to account for nondeterminism in heuristics.
- Baselines: standard recursive DFS (deterministic), a simple greedy local search, and a randomized hill-climber.
- Metrics: solution cost (problem-specific objective), wall-clock runtime, and success rate within fixed time budgets.

Key findings:
- **Optimality:** DFLS2-opt consistently found lower-cost solutions; averaged across instance families, improvements ranged from 10% (dense random graphs) to 35% (structured route networks) relative to DFS.
- **Runtime:** For comparable solution quality, DFLS2-opt reduced runtime by 20–40% against the greedy local search baseline, primarily due to reduced redundant exploration and early pruning.
- **Scalability:** DFLS2-opt’s wall-clock growth behaved roughly linear in practice for the tested instance sizes, while plain DFS exhibited super-linear/exponential trends on the largest instances.
- **Robustness:** Variability (standard deviation across runs) of results was lower for DFLS2-opt compared to randomized baselines, reflecting increased stability due to guided exploration.

We provide full experimental logs, raw outputs, and analysis notebooks in the repository to allow independent verification.

To further illustrate the performance and technical depth of DFLS2-opt, we elaborate on the following aspects:

## Benchmarking Methodology

Experiments were conducted on a workstation with 16GB RAM and a quad-core CPU. Each benchmark instance was run 30 times, and results were aggregated for mean, median, and standard deviation. The following parameters were varied:
- Instance size (number of nodes/cities)
- Graph density and topology
- Heuristic scoring function (greedy delta, degree-based, random)
- Pruning aggressiveness (threshold schedule)

## Detailed Results

| Instance Type                | Nodes | Opt. Impr. | Runtime Red. | Robustness (Std Dev) | Success Rate |
|------------------------------|-------|------------|--------------|----------------------|--------------|
| Dense Random Graphs          | 1000  | 10%        | 22%          | 5%                   | 100%         |
| Route Networks (city)        | 500   | 28%        | 35%          | 3%                   | 97%          |
| Scheduling Benchmarks        | 200   | 15%        | 24%          | 7%                   | 95%          |

DFLS2-opt maintained high success rates and low variability, indicating reliable performance across diverse problem types. Full logs and scripts are available for reproducibility.

## Algorithmic Innovations

DFLS2-opt introduces several technical improvements over classical DFS and local search:
- **Dynamic Focus Region:** The search space is adaptively restricted to promising subregions, reducing wasted computation.
- **Adaptive Pruning:** Pruning thresholds are not static; they evolve based on observed solution quality and runtime statistics.
- **Modular Heuristic Integration:** Users can swap in custom scoring functions without modifying the core search logic, enabling rapid experimentation.
- **Statistical Feedback Loop:** The algorithm tracks improvement frequency and depth, using this data to adjust search parameters in real time.

## Project Context and Impact

DFLS2-opt is designed for open science and extensibility. The repository includes:
- Well-documented source code with modular design
- Example notebooks for visualization and analysis
- Scripts for automated benchmarking and result aggregation
- Comprehensive README and usage instructions

The project aims to bridge the gap between theoretical algorithm research and practical, reproducible optimization tools for the community. It is suitable for both academic research and applied industrial use.

# Applications

DFLS2-opt is intentionally general-purpose. Example application scenarios include:
- **Transportation & Logistics:** Rapid identification of near-optimal routes and multi-stop tours in city-scale networks, useful for last-mile delivery.
- **Scheduling & Resource Allocation:** Assigning tasks to constrained resources with time windows and penalties, e.g., manufacturing or cloud job scheduling.
- **Constraint Satisfaction:** Solving structured CSPs such as exam timetabling or assignment problems where partial solutions enable strong pruning.
- **Research & Hybrid Methods:** Serving as a modular core that researchers can embed within memetic algorithms, or combine with learned heuristics from reinforcement learning.

# Open Science, Availability, and Reproducibility

All sources, datasets, and experimental scripts are publicly available:
- GitHub repository: https://github.com/rudraneel93/DFLS2-opt
- Zenodo deposition (this release): https://doi.org/10.5281/zenodo.17017798

The repository contains detailed README instructions, environment setup (requirements.txt), unit tests, and notebooks to reproduce reported experiments. We encourage community review, issue reports, and pull requests. For citation, please use the Zenodo DOI.

# Limitations & Ethical Considerations

While DFLS2-opt improves many practical instances, it is not a panacea. Limitations include:
- Dependence on heuristic quality: poorly chosen heuristics may reduce performance relative to tuned stochastic methods.
- Problem-specific tuning: some parameter adjustment strategies require domain insight for best results.
- Resource constraints: for extremely large instances, memory usage from deep stacks may still be significant.

Ethical considerations are minimal for algorithmic research, but users should be mindful when applying optimization in safety-critical systems; proper validation is required.

# Conclusion & Future Work

DFLS2-opt offers a pragmatic hybrid approach that brings together depth-first traversal and adaptive local search to deliver reproducible, efficient, and extensible solutions for combinatorial optimization. Future directions include:
- Extending to multi-objective optimization and dynamic problems.
- Integrating learned heuristics via imitation or reinforcement learning.
- Scaling to distributed settings and GPU-accelerated evaluation where applicable.

We invite the community to validate, critique, and extend the work via the repository and Zenodo record.

# Acknowledgements & Contact

This research was developed independently. The author thanks early testers and community members who provided feedback via GitHub issues. For correspondence: Rudraneel Das — https://github.com/rudraneel93 — Email: rudraneel93@gmail.com

# References

- Knuth, D. E. (1997). The Art of Computer Programming, Volume 1: Fundamental Algorithms. Addison-Wesley.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. Science, 220(4598), 671–680.
- Hoos, H. H., & Stützle, T. (2004). Stochastic Local Search: Foundations and Applications. Elsevier.
- Journal/benchmark datasets and code are included in the GitHub repository for reproducibility.
