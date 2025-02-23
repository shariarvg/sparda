# SPARDA (Sparse Differences Analysis)
Sparda is a Python implementation of Principal Differences Analysis (PDA), based on the paper "Principal Differences Analysis: Interpretable Characterization of Differences between Distributions" by Jonas Mueller and Tommi Jaakkola. PDA provides a principled way to identify and interpret the key differences between two probability distributions, making it particularly useful for analyzing dataset shifts, domain adaptation, and scientific comparisons.

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

## Generate example datasets
```python
X = np.random.normal(0, 1, (100, 10))  # Sample dataset 1
Y = np.random.normal(1, 1, (100, 10))  # Sample dataset 2 with a shift
```
## Perform Principal Differences Analysis
Let's say you want to get the Earth Mover's Distance of the sliced projections of the two datasets
```python
B, beta, d, last_improvement_step = relax(X,Y, n_it = 50, nu = 0.3, T = 50, lam = 1, gamma = 0.3, verbose = False)
```
Here, beta is the vector projecting the datasets into one-D space, and d is the wasserstein distance between X @ beta and Y @ beta. 

## Visualize or analyze the results
One way to visualize how well this algorithm works is to start with data in R^1, pick a random vector in R^3, and multiply every observation by that point. Then, we can see how well this algorithm recovers the original distribution. The **compare_projections** method allows us to visualize the original 1d distributions vs. the sliced recovered distributions.

```python
X = np.random.normal(0, 1, 100)
Y = np.random.normal(2, 1, 100)
compare_projections(X, Y, dim=10, n_it=50, nu=0.3, T=20, lam=1, gamma=0.3, labels=None)
```
![Perfect recovery](img/recovery.png?raw=true)
You can control the number of dimensions that the image is projected into, and also add white noise, and you'll find that recovery becomes worse.
```python
X = np.random.normal(0, 1, 100)
Y = np.random.normal(2, 1, 100)
compare_projections(X, Y, dim=200, n_it=50, nu=0.3, T=20, lam=1, gamma=0.3, labels=None, add_noise = True)
```
![Imperfect recovery (amidst noise)](img/recovery_with_noise.png?raw=true)

## Contact for questions and concerns
This repo was **not written by the paper's original authors**. I implemented for the purpose of studying interpretable distribution shift within language. If you are confused about the implementation, find an inconsistency with the original paper's algorithms, or generally think something is wrong -- please submit a PR to master or contact me (s v 2 2 6 at duke dot edu).
