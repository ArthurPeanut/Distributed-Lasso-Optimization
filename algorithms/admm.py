"""
Algorithm: ADMM (Alternating Direction Method of Multipliers)
Date: 2022/6/5
Author: Lau Haoqing
Description:
    使用 ADMM 求解带 L1 正则化的优化问题，其中数据由多个从属节点生成，
    每个节点产生观测值 b = A @ x_true + e。ADMM 的 x 子问题利用预先分解的常数矩阵，
    y 子问题则通过 soft-thresholding 算子更新。
"""

import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(z, threshold):
    """
    Soft-thresholding 算子（即 L1 正则项的近端算子）：
        prox(z, threshold) = sign(z) * max(|z| - threshold, 0)
    """
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

def main():
    # 参数设置
    slave_num = 20        # 从属节点数
    A_dim = (10, 300)     # 每个从属节点矩阵 A 的尺寸
    x_dim = 300           # 变量 x 的维数
    x_sparsity = 5        # x 的非零元素个数
    e_dim = 10            # 噪声 e 的维数

    # 生成真实的稀疏解 x_true
    x_true = np.zeros(x_dim)
    nonzero_indices = np.random.choice(x_dim, x_sparsity, replace=False)
    x_true[nonzero_indices] = np.random.normal(0, 1, x_sparsity)
    x0 = x_true.copy()    # 保存原始真实解，用于后续对比

    # 为每个从属节点生成数据：A, 噪声 e, 并计算 b = A @ x_true + e
    A = np.array([np.random.normal(0, 1, A_dim) for _ in range(slave_num)])
    e = np.array([np.random.normal(0, 0.2, e_dim) for _ in range(slave_num)])
    b = np.array([A[i] @ x_true + e[i] for i in range(slave_num)])

    # ADMM 算法参数
    c = 0.005
    p = 0.01
    epsilon = 1e-5
    max_iter = 1000

    # 初始化 ADMM 变量
    x_k = np.zeros(x_dim)
    y_k = np.zeros(x_dim)
    v_k = np.zeros(x_dim)
    x_k_prev = np.zeros(x_dim)
    results = []

    # 预先计算各从属节点数据的聚合矩阵：
    # sum1 = sum_i (A[i].T @ A[i])，shape 为 (x_dim, x_dim)
    # sum2 = sum_i (A[i].T @ b[i])，shape 为 (x_dim,)
    sum1 = np.einsum('ijk,ijl->kl', A, A)
    sum2 = np.einsum('ijk,ij->k', A, b)

    # 预先计算 x 更新中的常数矩阵 M = sum1 + c*I，并求其逆
    M = sum1 + c * np.eye(x_dim)
    M_inv = np.linalg.inv(M)

    # ADMM 迭代求解
    k = 1
    while k < max_iter:
        # x 子问题：求解 (sum1 + cI) x = sum2 + c*y_k - v_k
        x_k = M_inv @ (sum2 + c * y_k - v_k)

        # y 子问题：使用 soft-thresholding 进行更新
        y_k = soft_threshold(x_k + v_k / c, p / c)

        # v 子问题（对偶变量更新）
        v_k = v_k + c * (x_k - y_k)

        results.append(x_k.copy())

        # 判断收敛性：若连续两次 x 更新的变化小于 epsilon，则退出
        if np.linalg.norm(x_k - x_k_prev) < epsilon:
            break
        x_k_prev = x_k.copy()
        k += 1

    opt_x = x_k  # ADMM 最终得到的解
    print("Number of iterations:", len(results))

    # 计算每次迭代中与 ADMM 最优解和真实解之间的距离
    dist_from_opt = [np.linalg.norm(opt_x - xi) for xi in results]
    dist_from_true = [np.linalg.norm(x0 - xi) for xi in results]

    # 绘制收敛曲线
    plt.figure()
    plt.title("ADMM")
    plt.plot(dist_from_opt, label="Distance from optimal value")
    plt.plot(dist_from_true, label="Distance from true value")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
