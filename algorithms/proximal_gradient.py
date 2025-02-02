"""
Algorithm: Proximal Gradient Method
Date: 2022/6/5
Author: Lau Haoqing
"""

import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(z, threshold):
    """
    Soft-thresholding 算子，即 L1 范数的近端算子：
        prox(z, threshold) = sign(z) * max(|z| - threshold, 0)
    """
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def main():
    # 参数设置
    slave_num = 20       # 从属节点数
    A_dim = (10, 300)    # 每个从属节点矩阵 A 的尺寸
    x_dim = 300          # 变量 x 的维数
    x_sparsity = 5       # x 的非零元素个数
    e_dim = 10           # 噪声 e 的维数

    # 生成真实的稀疏解 x_true
    x_true = np.zeros(x_dim)
    nonzero_indices = np.random.choice(x_dim, x_sparsity, replace=False)
    x_true[nonzero_indices] = np.random.normal(0, 1, x_sparsity)

    # 为每个从属节点生成数据：A, 噪声 e, 并计算 b = A @ x_true + e
    A = np.array([np.random.normal(0, 1, A_dim) for _ in range(slave_num)])
    e = np.array([np.random.normal(0, 0.2, e_dim) for _ in range(slave_num)])
    b = np.array([A[i] @ x_true + e[i] for i in range(slave_num)])

    # 算法参数
    alpha = 0.001
    p = 1
    epsilon = 1e-5
    max_iter = 1000
    threshold = alpha * p  # 近端算子参数

    # 初始化迭代变量
    x_k = np.zeros(x_dim)
    results = [x_k.copy()]  # 存储每次迭代结果

    # 迭代求解：Proximal Gradient Descent
    for _ in range(max_iter):
        # 计算梯度：梯度为 sum_i [ alpha * A[i].T @ (A[i] @ x_k - b[i]) ]
        # 利用向量化：先计算 Ax (形状: (slave_num, 10))，再计算 residuals
        Ax = np.tensordot(A, x_k, axes=([2], [0]))  # 对每个节点：A[i] @ x_k
        residuals = Ax - b  # 每个从属节点的残差
        # 利用 einsum 计算每个 A[i].T @ residuals[i] 并对所有节点求和
        grad = alpha * np.einsum('iab,ia->b', A, residuals)

        # 梯度下降后做 soft-thresholding 投影（近端更新）
        x_new = soft_threshold(x_k - grad, threshold)

        # 判断收敛性
        if np.linalg.norm(x_new - x_k) < epsilon:
            x_k = x_new
            results.append(x_k.copy())
            break

        x_k = x_new
        results.append(x_k.copy())

    opt_x = x_k  # 最优解

    # 计算每次迭代中，与最优解和真实解的距离
    dist_from_opt = [np.linalg.norm(opt_x - xi) for xi in results]
    dist_from_true = [np.linalg.norm(x_true - xi) for xi in results]

    # 绘制收敛曲线
    plt.figure()
    plt.title("Proximal Gradient Descent")
    plt.plot(dist_from_opt, label="Distance from optimal value")
    plt.plot(dist_from_true, label="Distance from true value")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
