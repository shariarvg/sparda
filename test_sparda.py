import numpy as np
import sys, os
from relax import relax
from scipy.stats import wasserstein_distance

def generate_distributions(mu1, mu2, n_samples=1000, seed=42):
    """Generates two different distributions of real numbers."""
    np.random.seed(seed)
    
    # Two different normal distributions
    sigma1 =1  # Mean and standard deviation for first distribution
    sigma2 = 1  # Mean and standard deviation for second distribution
    
    X = np.random.normal(mu1, sigma1, n_samples)
    Y = np.random.normal(mu2, sigma2, n_samples)

    # Compute true Wasserstein distance
    true_wdist = wasserstein_distance(X, Y)
    
    return X, Y, true_wdist

def map_to_high_dim(X, Y, d=3, seed=42):
    """Maps 1D data into d-dimensional space using the same random vector."""
    np.random.seed(seed)
    
    # Generate a random vector for mapping
    random_vector = np.random.randn(d)
    random_vector = random_vector / np.linalg.norm(random_vector) ## unit vector
    
    # Map each sample to high-dimension
    X_high = np.outer(X, random_vector)
    Y_high = np.outer(Y, random_vector)
    
    return X_high, Y_high

def test_suite():
    """Runs the test pipeline to compare true and sliced Wasserstein distances."""
    # Generate base 1D distributions
    for mu2 in range(0, 10, 2):
        print(f"Mu1: {0}, Mu2:{mu2}")
        X, Y, true_wdist = generate_distributions(0, mu2)
        print(f"True Wasserstein Distance (1D): {true_wdist:.4f}")

        # Map to high-dimensional space
        X_high, Y_high = map_to_high_dim(X, Y, d=10)

        # Compute sliced Wasserstein distance
        _, _, sliced_wdist = relax(X_high, Y_high)
        print(f"Sliced Wasserstein Distance (High Dim): {sliced_wdist:.4f}")

        # Sanity Check: SWD should approximate the true Wasserstein distance
        print(f"Difference between True and Sliced Wasserstein: {abs(true_wdist - sliced_wdist):.4f}")

# Run the test suite
test_suite()
