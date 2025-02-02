"""
Algorithm: Sub-Gradient Descent
Date: 2022/6/5
Author: Lau Haoqing 
Description:
    使用子梯度下降法求解带 L1 正则项的优化问题，其中数据由多个从属节点生成，
    每个节点产生观测值 b = A @ x_true + e。对于 L1 正则项，由于在 x = 0 处不可导，
    使用随机取值的子梯度。
"""

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
slave_num = 20          # 从属节点数
A_dim = (10, 300)       # 每个从属节点中矩阵 A 的尺寸
x_dim = 300             # 变量 x 的维数
x_sparsity = 5          # x 的非零元素个数
e_dim = 10              # 噪声 e 的维数

# 生成真实的稀疏解 x_true
x_true = np.zeros(x_dim)
nonzero_indices = np.random.choice(x_dim, x_sparsity, replace=False)
x_true[nonzero_indices] = np.random.normal(0, 1, x_sparsity)

# 利用各从属节点生成数据：A, 噪声 e, 以及观测值 b = A @ x_true + e
A = np.array([np.random.normal(0, 1, A_dim) for _ in range(slave_num)])
e = np.array([np.random.normal(0, 0.2, e_dim) for _ in range(slave_num)])
# 利用 np.matmul 可直接计算形状为 (slave_num, 10) 的 A @ x_true
b = np.matmul(A, x_true) + e

# 子梯度下降算法参数
alpha_0 = 0.001
p = 1
epsilon = 1e-5

# 初始化变量
x_k = np.zeros(x_dim)       # 优化变量的初始值
x_ground_truth = x_true.copy()  # 保存真实解，用于后续对比
x_k_prev = np.zeros(x_dim)
results = []                # 用于保存每次迭代的 x_k

# 迭代计数器（可从较大值开始，以避免步长过大）
k = 10

while k < 1e3:
    # 计算数据项的梯度贡献：
    # 对每个从属节点 i， residual_i = A_i @ x_k - b_i, 最终求和 grad_data = sum_i A_i.T @ residual_i
    residuals = np.matmul(A, x_k) - b         # shape: (slave_num, 10)
    grad_data = np.einsum('ijk,ij->k', A, residuals)  # shape: (x_dim,)

    # 计算 L1 正则项的子梯度
    # 对于非零分量取 sign(x_k)；对于零分量，每个从属节点生成一个随机数（均匀分布在 [-1, 1]）
    nonzero_mask = (x_k != 0)
    # 为零分量生成 shape (slave_num, x_dim) 的随机数
    random_vals = np.random.uniform(-1, 1, size=(slave_num, x_dim))
    # 对每个从属节点构造其 L1 子梯度
    # 对非零位置直接采用 np.sign(x_k)（广播到各从属节点），零位置则用对应随机数
    l1_subgrad_all = np.where(nonzero_mask, np.sign(x_k), random_vals)
    # 累加各从属节点的 L1 子梯度
    l1_subgrad_sum = np.sum(l1_subgrad_all, axis=0)

    # 总子梯度：数据项梯度 + 正则项 p 乘以 L1 子梯度之和
    subgradient = grad_data + p * l1_subgrad_sum

    # 更新步长（衰减步长）
    alpha = alpha_0 / np.sqrt(k)
    # 更新 x_k（注意这里对 slave_num 求平均）
    x_k = x_k - (alpha / slave_num) * subgradient

    # 判断收敛性
    if np.linalg.norm(x_k - x_k_prev) < epsilon:
        results.append(x_k.copy())
        break
    else:
        x_k_prev = x_k.copy()
        results.append(x_k_prev.copy())
        k += 1

opt_x = x_k

# 计算每次迭代中与最优解和真实解的距离
dist_from_opt = [np.linalg.norm(opt_x - xi) for xi in results]
dist_from_true = [np.linalg.norm(x_ground_truth - xi) for xi in results]

# 绘制收敛曲线
plt.figure()
plt.title('Sub-Gradient Descent')
plt.plot(dist_from_opt, label='Distance from optimal value')
plt.plot(dist_from_true, label='Distance from true value')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.show()
