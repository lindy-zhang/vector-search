import numpy as np

def generate_vectors(n: int, d: int, seed: int=0) -> np.ndarray:
    # Create random num generator w/ numpy
    # Want same "random" vectors, use seed
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, d)) # Use normal/Gaussian distribution
    return vectors

if __name__ == "__main__":
    vectors = generate_vectors(n=1000, d=128)
    print("shape: ", vectors.shape)
    print("dtype: ", vectors.dtype)

    # Double check reproducibility: same seed gives identical results
    vectors2 = generate_vectors(n=1000, d=128)
    assert np.array_equal(vectors, vectors2), "same seed should produce identical vectors"
    print("reproducibility check: OK")