# Dynamic Focused Local Search (DFLS) TSP with 2-opt
def tsp_dfls_2opt(points, max_iter=10000, focus_radius=200):
    n = len(points)
    # Start with greedy DFS tour
    tour, tour_length = tsp_greedy_dfs(points)
    def dist(i, j):
        return euclidean_distance(points[i], points[j])
    # Precompute all distances for fast lookup
    D = [[dist(i, j) for j in range(n)] for i in range(n)]

    # Helper: compute cost of swapped segment only
    def delta_2opt(tour, i, j):
        a, b = tour[i], tour[i+1]
        c, d = tour[j], tour[(j+1)%n]
        # Remove edges (a-b) and (c-d), add (a-c) and (b-d)
        return (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])

    total_cost = sum(D[tour[k]][tour[k+1]] for k in range(n)) + D[tour[-1]][tour[0]]
    improved = True
    iter_count = 0
    max_iter = min(max_iter, 1000)  # Limit iterations for speed
    while improved and iter_count < max_iter:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Focus region: only consider swaps if cities are close
                if D[tour[i]][tour[j]] > focus_radius:
                    continue
                # 2-opt swap
                delta = delta_2opt(tour, i, j)
                if delta < 0:
                    # Perform swap
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    total_cost += delta
                    improved = True
                    break
            if improved:
                break
        iter_count += 1
    return tour, total_cost
import random
import math
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem, christofides

# Import DSP_TSP from dsp_tsp.py
from dsp_tsp import DSP_TSP

# Greedy TSP solver using DFS
def tsp_greedy_dfs(points):
    n = len(points)
    visited = [False] * n
    tour = []
    total_dist = 0.0
    current_idx = 0  # Start at the first city
    tour.append(current_idx)
    visited[current_idx] = True
    dfs = DynamicFocusSearch(points)
    for _ in range(n - 1):
        # Find nearest unvisited city
        best_dist = float('inf')
        best_idx = None
        for i in range(n):
            if visited[i]:
                continue
            d = euclidean_distance(points[current_idx], points[i])
            if d < best_dist:
                best_dist = d
                best_idx = i
        tour.append(best_idx)
        visited[best_idx] = True
        total_dist += best_dist
        current_idx = best_idx
    # Return to start
    total_dist += euclidean_distance(points[current_idx], points[tour[0]])
    tour.append(tour[0])
    return tour, total_dist

# Utility: Euclidean distance
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
def create_nx_graph(points):
    G = nx.Graph()
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                G.add_edge(i, j, weight=euclidean_distance(p1, p2))
    return G

def tsp_christofides(points):
    G = create_nx_graph(points)
    tour = christofides(G, weight='weight')
    # Ensure tour is a list of indices
    if isinstance(tour, list):
        pass
    else:
        tour = list(tour)
    # Ensure closed tour
    if tour[0] != tour[-1]:
        tour.append(tour[0])
    # Compute length
    length = sum(euclidean_distance(points[tour[i]], points[tour[i+1]]) for i in range(len(tour)-1))
    return tour, length


# Utility: Can we skip this point?
def can_skip(point, query, best_dist_so_far):
    for j in range(len(point)):
        if abs(point[j] - query[j]) > best_dist_so_far:
            return True
    return False

class DynamicFocusSearch:
    def __init__(self, data_points):
        self.data = data_points
        self.n = len(data_points)
        self.prev_best_index = None

    def query(self, query_point):
        best_dist = float('inf')
        best_idx = None
        # Focused start
        if self.prev_best_index is not None:
            p = self.data[self.prev_best_index]
            best_dist = euclidean_distance(query_point, p)
            best_idx = self.prev_best_index
        # Smart linear scan
        for i in range(self.n):
            if can_skip(self.data[i], query_point, best_dist):
                continue
            d = euclidean_distance(query_point, self.data[i])
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.prev_best_index = best_idx
        return best_idx, best_dist

# Naive linear scan for baseline
class NaiveLinearScan:
    def __init__(self, data_points):
        self.data = data_points
        self.n = len(data_points)
    def query(self, query_point):
        best_dist = float('inf')
        best_idx = None
        for i in range(self.n):
            d = euclidean_distance(query_point, self.data[i])
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

# Generate random data points
def generate_points(n, dim):
    return [[random.uniform(-1000, 1000) for _ in range(dim)] for _ in range(n)]

# Generate focused query sequence (temporal locality)
def generate_focused_queries(start, num, step_size):
    queries = [start[:]]
    for _ in range(num - 1):
        next_query = [x + random.uniform(-step_size, step_size) for x in queries[-1]]
        queries.append(next_query)
    return queries

# Generate random query sequence
def generate_random_queries(num, dim):
    return [[random.uniform(-1000, 1000) for _ in range(dim)] for _ in range(num)]

# Benchmarking
def benchmark(algorithm, queries):
    start = time.time()
    results = [algorithm.query(q) for q in queries]
    end = time.time()
    return results, end - start

if __name__ == "__main__":
    # Set dimension before any TSP or DFLS code
    dim = 2  # Set to 2 for TSP visualization
    n_points = 100000
    n_queries = 1000
    step_size = 5.0

    # DFLS TSP demo for n=100
    print("\nSolving TSP with DFLS 2-opt algorithm (n=100)...")
    dfls_points = generate_points(100, dim)
    dfls_tour, dfls_length = tsp_dfls_2opt(dfls_points, max_iter=5000, focus_radius=300)
    print(f"DFLS TSP tour length: {dfls_length:.2f}")
    print(f"DFLS TSP tour: {dfls_tour}")

    # Visualize DFLS TSP tour (for 2D only)
    if dim == 2:
        dfls_x = [dfls_points[i][0] for i in dfls_tour]
        dfls_y = [dfls_points[i][1] for i in dfls_tour]
        plt.figure(figsize=(7, 7))
        plt.plot(dfls_x, dfls_y, marker='o', color='orange', linewidth=2)
        plt.scatter(dfls_x, dfls_y, color='brown')
        for idx, (x, y) in enumerate(zip(dfls_x, dfls_y)):
            plt.text(x, y, str(idx), fontsize=8, ha='right')
        plt.title('DFLS 2-opt TSP Tour (n=100)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("Generating data points...")
    data = generate_points(n_points, dim)
    print("Generating query sequences...")
    focused_queries = generate_focused_queries([0.0]*dim, n_queries, step_size)
    random_queries = generate_random_queries(n_queries, dim)

    print("Benchmarking Naive Linear Scan (random queries)...")
    naive = NaiveLinearScan(data)
    naive_results_rand, naive_time_rand = benchmark(naive, random_queries)
    print(f"Naive Linear Scan (random): {naive_time_rand:.3f} seconds")

    print("Benchmarking DFS (random queries)...")
    dfs = DynamicFocusSearch(data)
    dfs_results_rand, dfs_time_rand = benchmark(dfs, random_queries)
    print(f"DFS (random): {dfs_time_rand:.3f} seconds")

    print("Benchmarking Naive Linear Scan (focused queries)...")
    naive = NaiveLinearScan(data)
    naive_results_focus, naive_time_focus = benchmark(naive, focused_queries)
    print(f"Naive Linear Scan (focused): {naive_time_focus:.3f} seconds")

    print("Benchmarking DFS (focused queries)...")
    dfs = DynamicFocusSearch(data)
    dfs_results_focus, dfs_time_focus = benchmark(dfs, focused_queries)
    print(f"DFS (focused): {dfs_time_focus:.3f} seconds")

    # Correctness check
    print("Checking correctness...")
    assert all(nr[0] == dr[0] for nr, dr in zip(naive_results_rand, dfs_results_rand)), "Mismatch in random queries!"
    assert all(nr[0] == dr[0] for nr, dr in zip(naive_results_focus, dfs_results_focus)), "Mismatch in focused queries!"
    print("All results match. DFS is correct.")

    print("Done.")

    # Visualization
    labels = ['Naive (Random)', 'DFS (Random)', 'Naive (Focused)', 'DFS (Focused)']
    times = [naive_time_rand, dfs_time_rand, naive_time_focus, dfs_time_focus]
    colors = ['#d62728', '#1f77b4', '#d62728', '#1f77b4']

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=colors)
    plt.ylabel('Time (seconds)')
    plt.title('DFS vs. Naive Linear Scan Benchmark')
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f'{v:.2f}s', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # TSP demo for 100 cities
    print("\nSolving TSP with DFS greedy heuristic (n=100)...")
    tsp_points = generate_points(100, dim)
    tour, tour_length = tsp_greedy_dfs(tsp_points)
    print(f"Greedy DFS TSP tour length: {tour_length:.2f}")
    print(f"Greedy DFS TSP tour: {tour}")

    print("\nSolving TSP with DFLS 2-opt algorithm (n=100)...")
    dfls_points = tsp_points  # Use the same points for fair comparison
    dfls_tour, dfls_length = tsp_dfls_2opt(dfls_points, max_iter=5000, focus_radius=300)
    print(f"DFLS TSP tour length: {dfls_length:.2f}")
    print(f"DFLS TSP tour: {dfls_tour}")

    # Side-by-side visualization for 100 cities (2D only)
    if dim == 2:
        dfs_x = [tsp_points[i][0] for i in tour]
        dfs_y = [tsp_points[i][1] for i in tour]
        dfls_x = [dfls_points[i][0] for i in dfls_tour]
        dfls_y = [dfls_points[i][1] for i in dfls_tour]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Greedy DFS plot
        axes[0].plot(dfs_x, dfs_y, marker='o', color='purple', linewidth=2)
        axes[0].scatter(dfs_x, dfs_y, color='red')
        for idx, (x, y) in enumerate(zip(dfs_x, dfs_y)):
            axes[0].text(x, y, str(idx), fontsize=8, ha='right')
        axes[0].set_title('Greedy DFS TSP Tour (n=100)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].grid(True)

        # DFLS 2-opt plot
        axes[1].plot(dfls_x, dfls_y, marker='o', color='orange', linewidth=2)
        axes[1].scatter(dfls_x, dfls_y, color='brown')
        for idx, (x, y) in enumerate(zip(dfls_x, dfls_y)):
            axes[1].text(x, y, str(idx), fontsize=8, ha='right')
        axes[1].set_title('DFLS 2-opt TSP Tour (n=100)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    # Benchmark for n=50, 200, 500, 1000, 2000, 5000 with visualizations
    tsp_sizes = [50, 200, 500, 1000, 2000, 5000]
    results = []
    for n_cities in tsp_sizes:
        print(f"\nBenchmarking TSP algorithms for n={n_cities}...")
        points = generate_points(n_cities, dim)

        # Greedy DFS
        start_time = time.time()
        tour_greedy, length_greedy = tsp_greedy_dfs(points)
        time_greedy = time.time() - start_time
        print(f"Greedy DFS: length={length_greedy:.2f}, time={time_greedy:.2f}s")

        # DFLS 2-opt
        start_time = time.time()
        tour_dfls, length_dfls = tsp_dfls_2opt(points, max_iter=5000, focus_radius=300)
        time_dfls = time.time() - start_time
        print(f"DFLS 2-opt: length={length_dfls:.2f}, time={time_dfls:.2f}s")

        # Christofides (skip for n > 1000 for speed)
        if n_cities <= 1000:
            start_time = time.time()
            tour_christo, length_christo = tsp_christofides(points)
            time_christo = time.time() - start_time
            print(f"Christofides: length={length_christo:.2f}, time={time_christo:.2f}s")
        else:
            tour_christo, length_christo, time_christo = None, None, None

        results.append({
            'n': n_cities,
            'greedy_length': length_greedy,
            'greedy_time': time_greedy,
            'dfls_length': length_dfls,
            'dfls_time': time_dfls,
            'christo_length': length_christo,
            'christo_time': time_christo
        })

        # Visualization for each size (2D only)
        if dim == 2:
            greedy_x = [points[i][0] for i in tour_greedy]
            greedy_y = [points[i][1] for i in tour_greedy]
            dfls_x = [points[i][0] for i in tour_dfls]
            dfls_y = [points[i][1] for i in tour_dfls]

            if n_cities <= 1000:
                christo_x = [points[i][0] for i in tour_christo] if tour_christo else None
                christo_y = [points[i][1] for i in tour_christo] if tour_christo else None
                fig, axes = plt.subplots(1, 3, figsize=(21, 7))
                # Greedy DFS plot
                axes[0].plot(greedy_x, greedy_y, marker='o', color='purple', linewidth=2)
                axes[0].scatter(greedy_x, greedy_y, color='red')
                axes[0].set_title(f'Greedy DFS TSP Tour (n={n_cities})\nLength: {length_greedy:.2f}, Time: {time_greedy:.2f}s')
                axes[0].set_xlabel('X')
                axes[0].set_ylabel('Y')
                axes[0].grid(True)
                # DFLS 2-opt plot
                axes[1].plot(dfls_x, dfls_y, marker='o', color='orange', linewidth=2)
                axes[1].scatter(dfls_x, dfls_y, color='brown')
                axes[1].set_title(f'DFLS 2-opt TSP Tour (n={n_cities})\nLength: {length_dfls:.2f}, Time: {time_dfls:.2f}s')
                axes[1].set_xlabel('X')
                axes[1].set_ylabel('Y')
                axes[1].grid(True)
                # Christofides plot (only if available)
                if christo_x and christo_y:
                    axes[2].plot(christo_x, christo_y, marker='o', color='blue', linewidth=2)
                    axes[2].scatter(christo_x, christo_y, color='cyan')
                    axes[2].set_title(f'Christofides TSP Tour (n={n_cities})\nLength: {length_christo:.2f}, Time: {time_christo:.2f}s')
                    axes[2].set_xlabel('X')
                    axes[2].set_ylabel('Y')
                    axes[2].grid(True)
                else:
                    axes[2].set_title('Christofides skipped (n > 1000)')
                plt.tight_layout()
                plt.show()
            elif n_cities > 1000:
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                # Greedy DFS plot
                axes[0].plot(greedy_x, greedy_y, marker='o', color='purple', linewidth=2)
                axes[0].scatter(greedy_x, greedy_y, color='red')
                axes[0].set_title(f'Greedy DFS TSP Tour (n={n_cities})\nLength: {length_greedy:.2f}, Time: {time_greedy:.2f}s')
                axes[0].set_xlabel('X')
                axes[0].set_ylabel('Y')
                axes[0].grid(True)
                # DFLS 2-opt plot
                axes[1].plot(dfls_x, dfls_y, marker='o', color='orange', linewidth=2)
                axes[1].scatter(dfls_x, dfls_y, color='brown')
                axes[1].set_title(f'DFLS 2-opt TSP Tour (n={n_cities})\nLength: {length_dfls:.2f}, Time: {time_dfls:.2f}s')
                axes[1].set_xlabel('X')
                axes[1].set_ylabel('Y')
                axes[1].grid(True)
                plt.tight_layout()
                plt.show()
            plt.show()

    # Additional visualizations: Tour length vs. runtime for each algorithm and city size

    sizes = [r['n'] for r in results]
    greedy_lengths = [r['greedy_length'] for r in results]
    dfls_lengths = [r['dfls_length'] for r in results]
    christo_lengths = [r['christo_length'] for r in results if r['christo_length'] is not None]
    greedy_times = [r['greedy_time'] for r in results]
    dfls_times = [r['dfls_time'] for r in results]
    christo_times = [r['christo_time'] for r in results if r['christo_time'] is not None]

    # Tour length vs. runtime scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(greedy_times, greedy_lengths, color='purple', label='Greedy DFS', s=100)
    plt.scatter(dfls_times, dfls_lengths, color='orange', label='DFLS 2-opt', s=100)
    if christo_times and christo_lengths:
        plt.scatter(christo_times, christo_lengths, color='blue', label='Christofides', s=100)
    for i, n in enumerate(sizes):
        plt.text(greedy_times[i], greedy_lengths[i], f'n={n}', fontsize=9, color='purple', ha='right')
        plt.text(dfls_times[i], dfls_lengths[i], f'n={n}', fontsize=9, color='orange', ha='right')
        if i < len(christo_times) and i < len(christo_lengths):
            plt.text(christo_times[i], christo_lengths[i], f'n={sizes[i]}', fontsize=9, color='blue', ha='right')
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Tour Length')
    plt.title('Tour Length vs. Runtime for TSP Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Bar chart comparing tour lengths for all algorithms and sizes
    import numpy as np
    x = np.arange(len(sizes))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, greedy_lengths, width, color='purple', label='Greedy DFS')
    plt.bar(x, dfls_lengths, width, color='orange', label='DFLS 2-opt')
    # Christofides bars only for sizes where data exists
    if christo_lengths:
        x_christo = np.arange(len(christo_lengths))
        plt.bar(x_christo + width, christo_lengths, width, color='blue', label='Christofides')
    plt.xticks(x, [f'n={n}' for n in sizes])
    plt.ylabel('Tour Length')
    plt.title('TSP Tour Length Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar chart comparing runtimes for all algorithms and sizes
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, greedy_times, width, color='purple', label='Greedy DFS')
    plt.bar(x, dfls_times, width, color='orange', label='DFLS 2-opt')
    if christo_times:
        x_christo = np.arange(len(christo_times))
        plt.bar(x_christo + width, christo_times, width, color='blue', label='Christofides')
    plt.xticks(x, [f'n={n}' for n in sizes])
    plt.ylabel('Runtime (seconds)')
    plt.title('TSP Algorithm Runtime Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\nTSP Benchmark Results:")
    print("| n | Greedy DFS Length | Greedy DFS Time (s) | DFLS 2-opt Length | DFLS 2-opt Time (s) | Christofides Length | Christofides Time (s) |")
    print("|---|-------------------|---------------------|-------------------|---------------------|---------------------|-----------------------|")
    for r in results:
        christo_length_str = f"{r['christo_length']:.2f}" if r['christo_length'] is not None else "-"
        christo_time_str = f"{r['christo_time']:.2f}" if r['christo_time'] is not None else "-"
        print(f"| {r['n']} | {r['greedy_length']:.2f} | {r['greedy_time']:.2f} | {r['dfls_length']:.2f} | {r['dfls_time']:.2f} | {christo_length_str} | {christo_time_str} |")
