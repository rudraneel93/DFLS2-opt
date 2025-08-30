import math
import itertools
from collections import defaultdict

# Dynamic State Pruning (DSP) TSP Algorithm Prototype

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class DSP_TSP:
    def __init__(self, points):
        self.points = points
        self.n = len(points)
        self.dist = [[euclidean_distance(points[i], points[j]) for j in range(self.n)] for i in range(self.n)]
        self.subpath_cache = dict()  # (frozenset, end) -> (cost, path)
        self.global_best = float('inf')
        self.global_tour = None

    def solve(self):
        # Start with all possible subpaths of length 2
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.subpath_cache[(frozenset([i, j]), j)] = (self.dist[i][j], [i, j])
        # Expand subpaths
        for size in range(3, self.n + 1):
            new_cache = dict()
            for subset in itertools.combinations(range(self.n), size):
                subset_set = frozenset(subset)
                for end in subset:
                    prev_subset = subset_set - {end}
                    best_cost = float('inf')
                    best_path = None
                    for prev_end in prev_subset:
                        if (prev_subset, prev_end) in self.subpath_cache:
                            prev_cost, prev_path = self.subpath_cache[(prev_subset, prev_end)]
                            cost = prev_cost + self.dist[prev_end][end]
                            # Geometric pruning: triangle inequality
                            if cost >= self.global_best:
                                continue
                            if cost < best_cost:
                                best_cost = cost
                                best_path = prev_path + [end]
                    if best_path:
                        new_cache[(subset_set, end)] = (best_cost, best_path)
            self.subpath_cache = new_cache
        # Complete tour
        for end in range(self.n):
            if (frozenset(range(self.n)), end) in self.subpath_cache:
                cost, path = self.subpath_cache[(frozenset(range(self.n)), end)]
                cost += self.dist[end][path[0]]  # Return to start
                if cost < self.global_best:
                    self.global_best = cost
                    self.global_tour = path + [path[0]]
        return self.global_tour, self.global_best

if __name__ == "__main__":
    # Example usage with 10 cities
    import random
    n_cities = 10
    dim = 2
    points = [[random.uniform(-1000, 1000) for _ in range(dim)] for _ in range(n_cities)]
    dsp = DSP_TSP(points)
    tour, cost = dsp.solve()
    print(f"DSP TSP shortest tour: {tour}")
    print(f"Tour length: {cost:.2f}")
