import numpy as np
import matplotlib.pyplot as plt
from relax import relax
from scipy.stats import wasserstein_distance

def plot_projected_datasets(X, Y, n_it=50, nu=0.3, T=20, lam=1, gamma=0.3, labels=None):
    """
    Projects two datasets using the relax algorithm and plots their 1D projections.
    
    Parameters:
    X, Y: numpy arrays of shape (n_samples, n_features)
    n_it: number of iterations for relax algorithm
    nu, T, lam, gamma: parameters for relax algorithm
    labels: tuple of strings (label_X, label_Y) for legend
    
    Returns:
    best_beta: the optimal projection vector
    best_distance: the Wasserstein distance between projected datasets
    """
    # Run relax algorithm to get optimal projection
    _, best_beta, best_distance, _ = relax(X, Y, n_it=n_it, nu=nu, T=T, lam=lam, gamma=gamma)
    
    # Project the datasets
    X_proj = X @ best_beta
    Y_proj = Y @ best_beta
    
    # Create plot
    plt.figure(figsize=(10, 5))
    
    # Plot histograms
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
        
    plt.hist(X_proj, bins=30, alpha=0.5, label=labels[0], density=True)
    plt.hist(Y_proj, bins=30, alpha=0.5, label=labels[1], density=True)
    
    plt.xlabel('Projected Value')
    plt.ylabel('Density')
    plt.title(f'Dataset Projections (Wasserstein Distance: {best_distance:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_beta, best_distance

def compare_projections(X1d, Y1d, dim=10, n_it=50, nu=0.3, T=20, lam=1, gamma=0.3, labels=None):
    """
    Takes two univariate datasets, projects them to higher dimensions, and compares original vs relaxed projections.
    
    Parameters:
    X1d, Y1d: numpy arrays of shape (n_samples,) - univariate datasets
    dim: dimension to project into (default: 10)
    n_it, nu, T, lam, gamma: parameters for relax algorithm
    labels: tuple of strings (label_X, label_Y) for legend
    
    Returns:
    projection_vector: random vector used for upward projection
    beta: optimal projection vector from relax algorithm
    distance: Wasserstein distance between final projections
    """
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
    
    # Ensure inputs are 1D arrays
    X1d = np.asarray(X1d).ravel()
    Y1d = np.asarray(Y1d).ravel()

    og_distance = wasserstein_distance(X1d, Y1d)
    
    # Create random projection vector (same for both datasets)
    np.random.seed(42)  # for reproducibility
    projection_vector = np.random.randn(dim)
    projection_vector = projection_vector / np.linalg.norm(projection_vector)  # normalize
    
    # Project to higher dimensions
    X_high = np.outer(X1d, projection_vector)  # Shape: (n_samples, dim)
    Y_high = np.outer(Y1d, projection_vector)  # Shape: (n_samples, dim)
    
    # Get optimal projection using relax
    _, beta, distance = relax(X_high, Y_high, n_it=n_it, nu=nu, T=T, lam=lam, gamma=gamma, verbose = False)

    # Project high-dimensional data back to 1D
    X_proj = X_high @ beta
    Y_proj = Y_high @ beta

    dist = wasserstein_distance(X_proj, Y_proj)
    print(X_proj - X1d)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot original distributions
    ax1.hist(X1d, bins=30, alpha=0.5, label=labels[0], density=True)
    ax1.hist(Y1d, bins=30, alpha=0.5, label=labels[1], density=True)
    ax1.set_title(f'Original Univariate Distributions\n(Wasserstein Distance: {og_distance:.4f})')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot projected distributions
    ax2.hist(X_proj, bins=30, alpha=0.5, label=f'{labels[0]} (projected)', density=True)
    ax2.hist(Y_proj, bins=30, alpha=0.5, label=f'{labels[1]} (projected)', density=True)
    ax2.set_title(f'Projected Distributions\n(Wasserstein Distance: {distance:.4f})')
    ax2.set_xlabel('Projected Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return projection_vector, beta, distance

