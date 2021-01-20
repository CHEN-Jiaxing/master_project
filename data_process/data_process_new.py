import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading

# 读取原始数据
yuanshi_shujv = pd.read_table('hld_001.fzp', encoding = 'gbk')
print(yuanshi_shujv)
# 提取有用数据
shuzhi_shujv_col = yuanshi_shujv.iloc[15:]

shuzhi_shujv_fenge = pd.DataFrame([jj.split(';') for jj in shuzhi_shujv_col['$VISION']])
shuzhi_shujv_fenge.columns = shuzhi_shujv_fenge.iloc[0]
shuzhi_shujv_fenge = shuzhi_shujv_fenge.iloc[1:]
print(shuzhi_shujv_fenge)
for i in range(7):
    print(shuzhi_shujv_fenge.iloc[0][i])
xx = shuzhi_shujv_fenge.iloc[0][6].split(' ')
print(xx)
