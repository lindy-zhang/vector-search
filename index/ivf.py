import numpy as np
from sklearn.cluster import KMeans
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from brute_force import euclidean_distances
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