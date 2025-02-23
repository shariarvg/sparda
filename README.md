# SPARDA (Sparse Differences Analysis)
Sparda is a Python implementation of Principal Differences Analysis (PDA), based on the paper "Principal Differences Analysis: Interpretable Characterization of Differences between Distributions" by Jonas Mueller. PDA provides a principled way to identify and interpret the key differences between two probability distributions, making it particularly useful for analyzing dataset shifts, domain adaptation, and scientific comparisons.

Features
Principal Differences Analysis (PDA) implementation
Highlights key differences between two distributions
Works with high-dimensional datasets
Based on eigen decomposition and optimization techniques from the original paper
Simple API for quick experimentation
Installation
You can install Sparda by doing the following:

```bash
git clone https://github.com/shariarvg/sparda.git
cd sparda
```

Usage
Here’s a basic example of how to use Sparda to analyze differences between two datasets:

python
Copy
Edit
import numpy as np
from sparda import PrincipalDifferencesAnalysis

## Generate example datasets
```python
X = np.random.normal(0, 1, (100, 10))  # Sample dataset 1
Y = np.random.normal(1, 1, (100, 10))  # Sample dataset 2 with a shift
```
## Perform Principal Differences Analysis
Let's say you want to get the Earth Mover's Distance of the sliced projections of the two datasets
```
B, beta, d, last_improvement_step = relax(X,Y, n_it = 50, nu = 0.3, T = 50, lam = 1, gamma = 0.3, verbose = False)
```
Here, beta is the vector projecting the datasets into one-D space, and d is the wasserstein distance between X @ beta and Y @ beta. 

## Visualize or analyze the results
One way to visualize how well this algorithm works is to start with data in R^1, pick a random vector in R^3, and multiply every observation by that point. Then, we can see how well this algorithm recovers the original distribution. The **compare_projections** method allows us to visualize the original 1d distributions vs. the sliced recovered distributions.

```
X = np.random.normal(0, 1, 100)
Y = np.random.normal(2, 1, 100)
compare_projections(X, Y, dim=10, n_it=50, nu=0.3, T=20, lam=1, gamma=0.3, labels=None)
```

How It Works
PDA finds directions in feature space that maximize differences between two distributions while suppressing variations within each dataset. Unlike traditional methods like PCA or CCA, which focus on variance or correlation, PDA explicitly targets inter-group differences.



Key Steps:
Compute Mean-Centered Covariances for both datasets
Solve the Generalized Eigenvalue Problem to find directions of maximum difference
Project Data onto these directions to visualize and quantify the differences
Applications
Dataset Shift Analysis: Identify changes in data distributions over time
Domain Adaptation: Understand differences between source and target domains
Feature Selection: Identify the most discriminative features
Scientific Data Analysis: Compare experimental vs. control groups
References
Jonas Mueller, "Principal Differences Analysis: Interpretable Characterization of Differences between Distributions," arXiv preprint arXiv:1510.08956, 2015. Paper Link
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you’d like to improve Sparda.
