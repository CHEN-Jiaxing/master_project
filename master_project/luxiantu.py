import numpy as np
from matplotlib import pyplot as plt

# 主干道
x_zgd = np.arange(-1,22)
y_zgd = np.zeros(len(x_zgd))
plt.plot(x_zgd, y_zgd, 'b')

# 区间起点
y_start = np.arange(-10,11) * 0.01
x_start = np.zeros(len(y_start))
plt.plot(x_start, y_start, 'k')

# 区间终点
y_end = np.arange(-10,11) * 0.01
x_end = np.zeros(len(y_end)) + 20
plt.plot(x_end, y_end, 'k')

# 第一处红绿灯
y_hld_1 = np.arange(-10,11) * 0.05
x_hld_1 = np.zeros(len(y_hld_1)) + 6
plt.plot(x_hld_1, y_hld_1, 'r')

# 第二处红绿灯
y_hld_2 = np.arange(-10,11) * 0.05
x_hld_2 = np.zeros(len(y_hld_2)) + 10
plt.plot(x_hld_2, y_hld_2, 'r')

# 第三处红绿灯
y_hld_3 = np.arange(-10,11) * 0.05
x_hld_3 = np.zeros(len(y_hld_3)) + 13
plt.plot(x_hld_3, y_hld_3, 'r')

# 第四处红绿灯
# y_hld_4 = np.arange(-10,11) * 0.05
# x_hld_4 = np.zeros(len(y_hld_4)) + 20
# plt.plot(x_hld_4, y_hld_4, 'r')

# 第一处人行道
y_rxd_1 = np.arange(-10,11) * 0.01
x_rxd_1 = np.zeros(len(y_rxd_1)) + 4
plt.plot(x_rxd_1, y_rxd_1, 'g')

# 第二处人行道
y_rxd_2 = np.arange(-10,11) * 0.01
x_rxd_2 = np.zeros(len(y_rxd_2)) + 8
plt.plot(x_rxd_2, y_rxd_2, 'g')

# 第三处人行道
y_rxd_3 = np.arange(-10,11) * 0.01
x_rxd_3 = np.zeros(len(y_rxd_3)) + 12
plt.plot(x_rxd_3, y_rxd_3, 'g')

# 第四处人行道
y_rxd_4 = np.arange(-10,11) * 0.01
x_rxd_4= np.zeros(len(y_rxd_4)) + 17
plt.plot(x_rxd_4, y_rxd_4, 'g')


plt.show()

