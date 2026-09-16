import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

def euclidean_distances(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    # Query q: (d, )
    # Database X: (n, d)
    # Want distance from q to every row in X: shape = (n, )
    diff = database - query # shape (n, d)
    distances = np.sqrt((diff**2).sum(axis=1))
    return distances

def brute_force_search(query: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    # Use euclidean_distances to find top-k nearest neighbors
    distances = euclidean_distances(query, database)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices

if __name__ == "__main__":
    # Test example: query closest to index 2, then 0, then 1
    database = np.array([
        [1.0, 1.0],   # index 0
        [10.0, 10.0], # index 1
        [0.0, 0.0],   # index 2
    ])
    query = np.array([0.1, 0.1])
    result = brute_force_search(query, database, k=2)
    print("nearest 2 indices:", result)
    assert list(result) == [2, 0], f"expected [2, 0], got {list(result)}"
    print("OK: matches hand-computed expected order")

    # Test at larger scale
    from synthetic_vectors import generate_vectors
    db = generate_vectors(n=10000, d=128, seed=0)
    q = generate_vectors(n=1, d=128, seed=1)[0]  # single query vector

    import time
    start = time.time()
    result = brute_force_search(q, db, k=10)
    elapsed = time.time() - start
    print(f"brute-force search over 10,000 vectors took {elapsed*1000:.2f} ms")
    print("top-10 indices:", result)

