#* **********************
#* Author: CHEN, Jiaxing
#* Function: 所有为工程起辅助作用或对比作用的函数
#* Date: 2021.11.04
#* Modified by:
#* Changes:
#* **********************

import numpy as np
import math

# ! Function: 整理1, 2车与 t 秒后的 3 车数据
def time_shift(data, shift_t):
    for i in range(len(data) - shift_t):
        cur_t = data[i, 0]
        v = -100
        a = -100

        for j in range(i, len(data)):
            if abs(data[j, 0] - (cur_t + shift_t)) < 0.1:
                v = data[j , 6]
                a = data[j , 7]
                break
        
        data[i, 6] = v
        data[i, 7] = a
    
    data = data[0 : int(len(data) - shift_t), :]
    for i in range(len(data)):
        if data[i, 6] == -100 and data[i, 7] == -100:
            data[i, :] = data[i-1, :]

    return data


# ! Function: 根据功率跟随策略计算总线需求下的燃料电池输出功率
# ! p_bus 总线功率、 p_fc_low 燃料电池最低输出功率、 p_fc_high 燃料电池最高输出功率

def pfc_calc(p_bus, p_fc_low = 3.19*2, p_fc_high = 25.82*2):
    p_fc = np.zeros(len(p_bus))
    for i in range(len(p_bus)):
        if p_bus[i] < p_fc_low:
            p_fc[i] = p_fc_low
        elif p_bus[i] > p_fc_high:
            p_fc[i] = p_fc_high
        else:
            p_fc[i] = p_bus[i]

    return p_fc

# ! 计算均方根误差
def rmse_calc(x, y):
    l = len(x)
    s = 0.0
    for i in range(l):
        s += (x[i]- y[i])**2
        
    return math.sqrt(s/l)