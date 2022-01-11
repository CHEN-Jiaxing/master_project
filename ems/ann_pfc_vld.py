#* **********************
#* Author: CHEN, Jiaxing
#* Function: validation for ann pfc prediction
#* Date: 2021.12.02
#* Modified by:
#* Changes:
#* **********************

# 画图中文显示
import matplotlib
matplotlib.rc("font",family='DengXian')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.models import load_model
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# ! 读取数据
data = np.load('./data_master/input_pfc_12.9/input_pfc_vld.npy')
# data = np.concatenate((data,data,data,data,data),axis=0)
data[:,0] = savgol_filter(data[:,0], 31, 3, mode= 'nearest')
data[:,1] = savgol_filter(data[:,1], 31, 3, mode= 'nearest')
data[:,2] = savgol_filter(data[:,2], 31, 3, mode= 'nearest')
data[:,3] = savgol_filter(data[:,3], 31, 3, mode= 'nearest')
data[:,4] = savgol_filter(data[:,4], 31, 3, mode= 'nearest')
data[:,5] = savgol_filter(data[:,5], 31, 3, mode= 'nearest')
data[:,6] = savgol_filter(data[:,6], 31, 3, mode= 'nearest')

# 网络参数设置
model = load_model("ann_pfc.h5")
BATCH_START = 0
TIME_STEPS = 15
BATCH_SIZE = 4
INPUT_SIZE = 6
OUTPUT_SIZE = 1

# 数据长度取整
BATCH_NUM = int(len(data) / (BATCH_SIZE * TIME_STEPS))
data = data[0:BATCH_NUM * BATCH_SIZE * TIME_STEPS]
print("当前数据总数:", len(data))

# 输入输出获取len(data) - BATCH_SIZE * TIME_STEPS
def batch():
    global BATCH_START, TIME_STEPS
    t = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE)
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 0:6]
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:7]
    BATCH_START += TIME_STEPS
    return [X_batch, Y_batch, t]

loss = []

# 实际功率和预测输出功率
y_pfc = np.array([])
p_pfc = np.array([])

for step in range(int((len(data) - BATCH_SIZE * TIME_STEPS)/TIME_STEPS) + 1):
    X_batch, Y_batch, t = batch()
    
    pred = model.predict(X_batch, BATCH_SIZE)
    
    plt.subplot(2,1,1)
    plt.title("功率预测情况", fontsize=12)
    plt.plot(t[:], Y_batch[: , 0], 'r',label="实际功率")
    plt.plot( t[:], pred[: , 0], 'g--',label="预测功率")
    plt.xlabel("时间(s)", fontsize=12)
    plt.ylabel("功率(kW)", fontsize=12)
    if step == 0:
        plt.legend()
    
    # y_pfc = np.append(y_pfc, Y_batch[: , 0])
    p_pfc = np.append(p_pfc, pred[: , 0])
    
    results = model.evaluate(X_batch, Y_batch, BATCH_SIZE)
    # print("test loss, test acc:", results)
    loss.append(results)

plt.figure()
plt.subplot(2,1,2)
loss = np.array(loss, dtype=np.float)
plt.plot(np.array(loss[:,0]))
# plt.xlim((0,30))
# plt.ylim((0,100))
plt.ylabel('误差')
plt.xlabel('批次')
plt.title('功率预测情况误差变化图——验证集')
plt.show()
    



# 计算均方根误差
def rmse_calc(x, y):
    l = len(x)
    s = 0.0
    for i in range(l):
        s += (x[i]- y[i])**2
    return math.sqrt(s/l)

# 均方根误差
rmse_pfc = rmse_calc(p_pfc, y_pfc)
print("功率均方根误差：", rmse_pfc)

# np.save("p_ann", p_pfc)

print("=====END=====")