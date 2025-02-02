# Distributed Lasso Optimization

## Course Information

This project was completed as part of the course **Optimization** in the **third year of the Computer Science program at Sun Yat-sen University**.


## Problem Statement

This project focuses on solving a **distributed L1-regularized least squares problem** (Lasso) in a network of 20 nodes. Each node \( i \) collects linear measurements modeled as:

\[
    b_i = A_i x + e_i
\]

where:

- \( b_i \) is a **10-dimensional observation vector**,
- \( A_i \) is a **\(10 \times 300\) measurement matrix**,
- \( x \) is a **300-dimensional unknown sparse vector** with a sparsity of 5,
- \( e_i \) is a **10-dimensional noise vector**.

The objective is to **recover \( x \) from all measurements \( b_i \) and matrices \( A_i \)** by solving the following **L1-regularized least squares problem**:

\[
    \min_{x} \frac{1}{2} \sum_{i=1}^{20} \| A_i x - b_i \|_2^2 + p \| x \|_1
\]

where \( p \geq 0 \) is a regularization parameter.


## Optimization Algorithms

To solve this problem in a **distributed setting**, the following optimization methods are implemented:

1. **Proximal Gradient Method**
2. **Alternating Direction Method of Multipliers (ADMM)**
3. **Subgradient Method**

Each algorithm is implemented in a way that allows computation to be **distributed across multiple nodes**, enabling efficient sparse recovery.

## Experimental Setup

- The nonzero elements of \( x \) follow a **Gaussian distribution** with mean 0 and variance 1.
- The elements of each matrix \( A_i \) are **Gaussian-distributed** with mean 0 and variance 1.
- The noise vector \( e_i \) follows a **Gaussian distribution** with mean 0 and variance 0.2.

## Repository Structure

```
📂 distributed-lasso-optimization
│── 📂 algorithms              # Implementation of optimization algorithms
│   ├── proximal_gradient.py   # Proximal Gradient Method
│   ├── admm.py                # ADMM implementation
│   ├── subgradient.py         # Subgradient Method
│
│── 📂 results                 # Experiment results and analysis
│   ├── plots/                 # Visualizations of convergence and performance
│
│── README.md                  
│── requirements.txt            
│── main.py                     # Main script to run experiments
```

## Installation

To run the experiments, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

To run the main script with default settings:

```bash
python main.py
```

To experiment with different optimization methods, modify the script arguments:

```bash
python main.py --method proximal_gradient
python main.py --method admm
python main.py --method subgradient
```

## Results & Analysis

The performance of the three optimization methods is evaluated based on:

- **Convergence speed**
- **Reconstruction accuracy**
- **Computational efficiency**

Plots and comparisons are stored in the `results/plots/` directory.

## References

- Boyd, S., Parikh, N., & Chu, E. (2011). *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.*
- Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization: A Basic Course.*
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
