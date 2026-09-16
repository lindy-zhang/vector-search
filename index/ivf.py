import numpy as np
from sklearn.cluster import KMeans
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from brute_force import euclidean_distances, brute_force_search
from collections import defaultdict

class IVFIndex:
    def __init__(self, nlist: int):
        # nlist = num of lists = num of clusters we'll split database into
        self.nlist = nlist
        self.clusters = KMeans(n_clusters=nlist)
        self.database = None
        self.cluster_to_indices = defaultdict(list)
    
    def build(self, database: np.ndarray):
        self.database = database
        self.clusters.fit(database)
        labels = self.clusters.labels_
    
        for i, cluster_id in enumerate(labels):
            self.cluster_to_indices[cluster_id].append(i)
    
    def search(self, query: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        # nprobe = how many clusters to search

        centroids = self.clusters.cluster_centers_
        centroid_distances = euclidean_distances(query=query, database=centroids)
        nearest_cluster_ids = np.argsort(centroid_distances)[:nprobe]

        # Gather all database indices from nearest_cluster_ids
        candidate_indices = []
        for cluster_id in nearest_cluster_ids:
            candidate_indices.extend(self.cluster_to_indices[cluster_id])
        candidate_indices = np.array(candidate_indices)

        candidate_vectors = self.database[candidate_indices]
        local_result = brute_force_search(query, candidate_vectors, k)
        global_result = candidate_indices[local_result]
        return global_result

if __name__ == "__main__":
    from synthetic_vectors import generate_vectors

    db = generate_vectors(n=1000, d=64, seed=0)
    index = IVFIndex(nlist=10)
    index.build(db)

    print("number of clusters:", len(index.cluster_to_indices))
    total_assigned = sum(len(v) for v in index.cluster_to_indices.values())
    print("total vectors assigned across all clusters:", total_assigned)
    assert total_assigned == 1000, "every vector should belong to exactly one cluster"

    cluster_sizes = [len(v) for v in index.cluster_to_indices.values()]
    print("cluster sizes:", sorted(cluster_sizes))

    # Test
    query = generate_vectors(n=1, d=64, seed=99)[0]

    ivf_result = index.search(query, k=10, nprobe=3)
    print("IVF top-10:", ivf_result)

    true_result = brute_force_search(query, db, k=10)
    print("brute-force top-10:", true_result)

    overlap = len(set(ivf_result) & set(true_result))
    print(f"overlap with true nearest neighbors: {overlap}/10")