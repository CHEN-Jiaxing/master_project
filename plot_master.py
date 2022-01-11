#* **********************
#* Author: CHEN, Jiaxing
#* Function: 车辆信息预测的神经网络输入输出画图
#* Date: 2021.04.06
#* Modified by:
#* Changes:
#* **********************

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='DengXian')
import pandas as pd
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

# ? va 网络输入整理
'''
data = np.load('./data_master/input_va_time_delay/input_1_va_pre.npy')
data[:,1]= savgol_filter(data[:,1], 11, 3, mode= 'nearest')
data[:,2]= savgol_filter(data[:,2], 11, 3, mode= 'nearest')
data[:,3]= savgol_filter(data[:,3], 11, 3, mode= 'nearest')
data[:,4]= savgol_filter(data[:,4], 11, 3, mode= 'nearest')
data[:,5]= savgol_filter(data[:,5], 11, 3, mode= 'nearest')
data[:,6]= savgol_filter(data[:,6], 11, 3, mode= 'nearest')
data[:,7]= savgol_filter(data[:,7], 11, 3, mode= 'nearest')
# 网络参数设置
BATCH_SIZE = 20
# 数据长度取整
BATCH_NUM = int(len(data) / BATCH_SIZE)
data = data[0:BATCH_NUM * BATCH_SIZE]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data[:1000,1])
plt.ylabel('位置(m)', fontsize = 12)
plt.xlabel('时间(s)', fontsize = 12)
plt.grid(True)
plt.title("前方第一辆车距离红绿灯的距离", fontsize= 12)
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data[:1000,2],'b',label="V1")
plt.plot(data[:1000,4],'r-.',label="V2")
plt.ylabel('速度(km/h)', fontsize = 12)
plt.xlabel('时间(s)', fontsize = 12)
plt.title("前方第一辆车与第二辆车的速度", fontsize= 12)
plt.grid(True)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(data[:1000,3],'b',label="a1")
plt.plot(data[:1000,5],'r-.',label="a2")
plt.ylabel('加速度(m/(s^2))', fontsize = 12)
plt.xlabel('时间(s)', fontsize = 12)
plt.grid(True)
plt.legend()
plt.title("前方第一辆车与第二辆车的加速度", fontsize= 12)
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(data[:1000,6],'r')
plt.ylabel('速度(km/h)', fontsize = 12)
plt.xlabel('时间(s)', fontsize = 12)
plt.title("当前行驶车辆的速度", fontsize= 12)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(data[:1000,7],'b--')
plt.ylabel('加速度(m/(s^2))', fontsize = 12)
plt.xlabel('时间(s)', fontsize = 12)
plt.title("当前行驶车辆的加速度", fontsize= 12)
plt.grid(True)
plt.show()
'''

# ? 预测速度加速度rmse误差
'''
# 预测时间分别为 1,5,10,15,20
plt.figure(10)
plt.subplot(2,1,1)
wucha_t = np.array([1,5,10,15,20,30])
wucha_v = np.array([42.60051933926142, 38.67402804118026,37.08230328345541,36.04958699164945,20.257797398446876,31.835052429957216])
wucha_a = np.array([1.8025755930847438, 1.8780114721239178,2.083795538735273,1.9968218542738514,2.6494643299751393,4.2131032564004665])
plt.plot(wucha_t,wucha_v,'b',label="速度RMSE")
plt.plot(wucha_t,wucha_a,'b-.',label="加速度RMSE")
plt.xlabel("预测时间", fontsize= 14)
plt.ylabel("RMSE", fontsize= 14)
plt.title("不同预测步长下的速度与加速度RMSE变化", fontsize= 14)
plt.legend(fontsize = 14)
plt.grid()
plt.show()
'''

# ? 燃料电池特性
'''
plt.figure(2)
fc_pwr_map = np.array([0,4800,9500,14100,18500,22700,26800,29700,32500])
fc_eff_map = np.array([0.00,0.55,0.525,0.515,0.500,0.495,0.49,0.48,0.47])
fc_fuel_map = np.array([0.0000001,0.0733,0.151,0.2307,0.3074,0.3812,0.4577,0.5166,0.5758])

#plt.subplot(2,2,1)
plt.plot(fc_pwr_map, fc_fuel_map)
plt.plot(fc_pwr_map, fc_fuel_map,'r.')
plt.ylabel("氢耗量(g/s)",fontsize=12)
plt.xlabel("燃料电池单堆净输出功率(W)"+"\n"+"a:燃料电池氢耗量曲线",fontsize=12)
plt.grid(True)
plt.figure(3)
#plt.subplot(2,2,3)
plt.plot(fc_pwr_map,fc_eff_map)
plt.plot(fc_pwr_map,fc_eff_map,'r.')
plt.ylabel("效率(%)",fontsize=12)
plt.xlabel("燃料电池单堆净输出功率(W)"+"\n"+"b:燃料电池效率曲线",fontsize=12)
plt.grid(True)
plt.show()
'''

# ? 蓄电池 soc-voc-r
'''
plt.figure(3)
ess_voc_map = np.array([3.438*96,3.533*96,3.597*96,3.627*96,3.658*96,3.719*96,3.785*96,3.858*96 ,3.946*96,4.049*96,4.18*96])
ess_soc_map = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ess_chr_map = np.array([0.16,0.15,0.13,0.13,0.13,0.13,0.13,0.13,0.13,0.13,0.13])
ess_dis_map = np.array([0.2,0.18,0.13,0.13,0.13,0.13,0.13,0.13,0.13,0.13,0.13])
plt.subplot(3,1,1)
plt.plot(ess_soc_map, ess_voc_map)
plt.plot(ess_soc_map, ess_voc_map,'r.')
plt.xlabel("蓄电池SOC")
plt.ylabel("蓄电池开路电压(V)")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(ess_soc_map, ess_chr_map)
plt.plot(ess_soc_map, ess_chr_map,'r.')
plt.xlabel("蓄电池SOC")
plt.ylabel("蓄电池充电电阻(ohm)")
plt.ylim((0,0.25))
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(ess_soc_map, ess_dis_map)
plt.plot(ess_soc_map, ess_dis_map,'r.')
plt.xlabel("蓄电池SOC")
plt.ylabel("蓄电池放电电阻(ohm)")
plt.ylim((0,0.25))
plt.grid(True)

plt.show()
'''



# ? 不同数量的隐藏层下的速度均方根误差值对比
'''
n_hd = np.array([5,6,7,8,9,10,11,12])
v1_h = np.array([11.33637417,10.66872749,9.271995437,10.10764292,8.934463334,9.348949581,9.855654218,9.862126862])
v5_h = np.array([10.70462996,10.60021982,9.41855576,10.31974562,9.108066316,9.283876806,9.585002398,9.514824355])
v15_h = np.array([20.06092646,19.64148358,19.37171649,19.88295346,18.68176737,19.09947246,19.34207594,18.69981423])
v30_h = np.array([17.24260328,18.28326464,16.25214515,17.03824742,16.20515867,15.92968645,15.56985686,17.00995055])
v45_h = np.array([21.25149478,19.50671979,19.71504167,18.96994146,18.03480573,19.91329929,18.22540837,17.73775159])
v60_h = np.array([19.91861378,20.02058618,19.41875962,20.765207,20.63324581,21.2635501,20.96601406,19.38112832])

plt.plot(n_hd,v1_h, 'r',label='1s')
plt.plot(n_hd,v5_h, 'g',label='5s')
plt.plot(n_hd,v15_h, 'b',label='15s')
plt.plot(n_hd,v30_h, 'y',label='30s')
plt.plot(n_hd,v45_h, 'c',label='45s')
plt.plot(n_hd,v60_h, 'm',label='60s')
plt.xlabel("隐藏层数量", fontsize= 12)
plt.ylabel("RMSE", fontsize= 12)
plt.legend()
plt.grid(True)
plt.title("不同数量的隐藏层下的速度均方根误差值对比")
plt.show()
'''

# ? 不同数量的隐藏层下的加速度均方根误差值对比
'''
n_hd = np.array([5,6,7,8,9,10,11,12])
a1_h = np.array([1.267944818,1.295353245,1.255031043,1.269569843,1.265477423,1.275872656,1.328985432,1.355357765])
a5_h = np.array([1.234456098,1.281363161,1.28871693,1.281505324,1.244016792,1.271487365,1.215744036,1.244819022])
a15_h = np.array([1.460679097,1.479665113,1.418155292,1.595172538,1.633653955,1.499160019,1.540466043,1.970397312])
a30_h = np.array([1.292176967,1.252209152,1.489645694,1.491374459,1.339578703,1.344869314,1.372501418,1.371709696])
a45_h = np.array([4.335655079,4.333411738,4.344827468,4.337706164,4.323834018,4.343289103,4.435780357,4.367419889])
a60_h = np.array([1.505935379,1.537040889,1.419478289,1.295261507,1.390877316,1.378672828,1.398922312,1.415475387])

plt.plot(n_hd,a1_h, 'r',label='1s')
plt.plot(n_hd,a5_h, 'g',label='5s')
plt.plot(n_hd,a15_h, 'b',label='15s')
plt.plot(n_hd,a30_h, 'y',label='30s')
plt.plot(n_hd,a45_h, 'c',label='45s')
plt.plot(n_hd,a60_h, 'm',label='60s')
plt.xlabel("隐藏层数量", fontsize= 12)
plt.ylabel("RMSE", fontsize= 12)
plt.legend()
plt.grid(True)
plt.title("不同数量的隐藏层下的加速度均方根误差值对比")
plt.show()
'''

# ? 不同的回溯步长下的速度均方根误差值对比
'''
n_rc = np.array([1,2,3,4,5,6,7,8,9,10])
v1_r = np.array([13.1298693, 11.17810144, 10.13621947, 10.24065607, 9.586050742, 9.818964008, 9.738000523, 8.975997806, 
7.670912184, 8.757645965])
v5_r = np.array([13.83363657, 9.87871819, 8.964293749, 9.775035845, 9.849895283, 9.989340636, 9.262288747, 9.381669616, 
8.148522072, 9.085250591])
v15_r = np.array([24.40755426, 23.38040433, 19.73762843, 18.51751448, 18.73560496, 17.6792398, 17.95784447, 17.63076942, 
17.70084357, 17.72785876])
v30_r = np.array([24.17950497, 16.48885742, 18.21086513, 15.46367248, 15.72111019, 15.44897457, 15.10418138, 16.02562119, 15.75992688, 14.51092707])
v45_r = np.array([24.49154843, 20.39769613, 18.94236811, 19.4504411, 18.15570783, 18.49525139, 18.54652066, 18.19403508, 
17.91595481, 17.10355483])
v60_r = np.array([19.19130166, 21.36624418, 20.48078445, 21.02916462, 21.60718327, 20.43703851, 20.09878487, 18.37401115, 21.09695548, 19.27741291])

plt.plot(n_rc,v1_r, 'r',label='1s')
plt.plot(n_rc,v5_r, 'g',label='5s')
plt.plot(n_rc,v15_r, 'b',label='15s')
plt.plot(n_rc,v30_r, 'y',label='30s')
plt.plot(n_rc,v45_r, 'c',label='45s')
plt.plot(n_rc,v60_r, 'm',label='60s')
plt.xlabel("回溯步长", fontsize= 12)
plt.ylabel("RMSE", fontsize= 12)
plt.legend()
plt.grid(True)
plt.title("不同回溯步长下的速度均方根误差值对比")
plt.show()
'''

# ? 不同的回溯步长下的加速度均方根误差值对比
'''
n_rc = np.array([1,2,3,4,5,6,7,8,9,10])
a1_r = np.array([1.335288591, 1.412859133, 1.31355789, 1.27414472, 1.244111547, 1.314878823, 1.348010036, 1.184129092, 1.227492574, 1.237517874])
a5_r = np.array([1.345822061, 1.374597314, 1.342522272, 1.225637959, 1.207646355, 1.245627965, 1.214451993, 1.21312749, 
1.208164829, 1.200037674])
a15_r = np.array([1.593852946, 1.499678145, 1.396355546, 1.370383935, 1.349156711, 1.364576197, 1.332313844, 1.389420277, 2.568636814, 1.882312295])
a30_r = np.array([1.271569397, 1.46785435, 1.644622103, 1.418278929, 1.347540207, 1.412554512, 1.344480561, 1.252324674, 
1.275860835, 1.257496187])
a45_r = np.array([4.240336256, 4.253335526, 4.288932935, 4.451609109, 4.319831986, 4.399253036, 4.350281789, 4.381269054, 4.415923903, 4.426631176])
a60_r = np.array([1.749376501, 1.269178752, 1.260382398, 1.480450694, 1.252001286, 1.765222209, 1.472012186, 1.441878833, 1.213030798, 1.273546226])

plt.plot(n_rc,a1_r, 'r',label='1s')
plt.plot(n_rc,a5_r, 'g',label='5s')
plt.plot(n_rc,a15_r, 'b',label='15s')
plt.plot(n_rc,a30_r, 'y',label='30s')
plt.plot(n_rc,a45_r, 'c',label='45s')
plt.plot(n_rc,a60_r, 'm',label='60s')
plt.xlabel("回溯步长", fontsize= 12)
plt.ylabel("RMSE", fontsize= 12)
plt.legend()
plt.grid(True)
plt.title("不同回溯步长下的加速度均方根误差值对比")
plt.show()
'''

# ? 预测速度加速度rmse误差
'''
# 预测时间分别为 1,5,10,15,20
plt.figure(10)

wucha_t = np.array([1,5,15,30,45,60])
wucha_v_1 = np.array([7.569341677,8.178187759,10.92659465,10.79199673,14.21511391,18.76672346])
wucha_a_1 = np.array([1.190375028,2.662551524,2.544254517,1.274940165,1.284782397,1.460022884])
plt.plot(wucha_t,wucha_v_1,'b',label="场景一速度RMSE")
plt.plot(wucha_t,wucha_a_1,'b-.',label="场景一加速度RMSE")
wucha_v_2 = np.array([9.842530444,10.55472943,12.10452098,12.2458669062273,19.51094397,19.61250271])
wucha_a_2 = np.array([1.320464424,1.327660245,1.34674576,1.352897815,1.625380205,1.663274637])
plt.plot(wucha_t,wucha_v_2,'r',label="场景二速度RMSE")
plt.plot(wucha_t,wucha_a_2,'r-.',label="场景二加速度RMSE")
plt.xlabel("预测时间", fontsize= 14)
plt.ylabel("RMSE", fontsize= 14)
plt.title("不同预测步长下的速度与加速度RMSE变化", fontsize= 14)
plt.legend(fontsize = 14)
plt.grid()
plt.show()
'''

# ? 不同的soc参数下的功率分配
'''
pbat_10 = np.load("./pbat_10_0.4.npy")
pbat_100 = np.load("./pbat_100_0.4.npy")
pbat_1000 = np.load("./pbat_1000_0.4.npy")

pfc_10 = np.load("./pfc_10_0.4.npy")
pfc_100 = np.load("./pfc_100_0.4.npy")
pfc_1000 = np.load("./pfc_1000_0.4.npy")

soc_10 = np.load("./soc_10_0.4.npy")
soc_100 = np.load("./soc_100_0.4.npy")
soc_1000 = np.load("./soc_1000_0.4.npy")

plt.figure()
plt.subplot(3,1,1)
plt.plot(pbat_10, 'r',label='beta参数为10')
plt.plot(pbat_100, 'g',label='beta参数为100')
plt.plot(pbat_1000, 'y',label='beta参数为1000')
plt.title('蓄电池输出功率')
plt.xlabel("时间")
plt.ylabel("输出功率")
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(pfc_10, 'r',label='beta参数为10')
plt.plot(pfc_100, 'g',label='beta参数为100')
plt.plot(pfc_1000, 'y',label='beta参数为1000')
plt.title('燃料电池输出功率')
plt.xlabel("时间")
plt.ylabel("输出功率")
plt.legend(bbox_to_anchor=(1, 1))
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(soc_10, 'r',label='beta参数为10')
plt.plot(soc_100, 'g',label='beta参数为100')
plt.plot(soc_1000, 'y',label='beta参数为1000')
plt.title('SOC轨迹变化')
plt.xlabel("时间")
plt.ylabel("SOC")
plt.legend(bbox_to_anchor=(1, 1))
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
'''

# ? 不同的初始soc
'''
data_1 = np.load('./data_master/output_dp_socs/input_01_0.4.npy')
data_2 = np.load('./data_master/output_dp_socs/input_01_0.5.npy')
data_3 = np.load('./data_master/output_dp_socs/input_01_0.6.npy')
data_4 = np.load('./data_master/output_dp_socs/input_01_0.7.npy')
plt.subplot(2,1,1)
plt.plot(data_1[:,1]/37/3.6, 'b', label = '初始SOC为0.4')
plt.plot(data_2[:,1]/37/3.6, 'r-', label = '初始SOC为0.5')
plt.plot(data_3[:,1]/37/3.6, 'g', label = '初始SOC为0.6')
plt.plot(data_4[:,1]/37/3.6, 'k-.', label = '初始SOC为0.7')
plt.xlabel('时间')
plt.ylabel('SOC')
plt.title('不同初始SOC下动态规划求解的轨迹')
plt.legend(bbox_to_anchor=(1, 0.48))
plt.subplot(2,1,2)
plt.plot(data_1[:,6]*1000, 'b', label = '初始SOC为0.4')
plt.plot(data_2[:,6]*1000, 'r-', label = '初始SOC为0.5')
plt.plot(data_3[:,6]*1000, 'g', label = '初始SOC为0.6')
plt.plot(data_4[:,6]*1000, 'k-.', label = '初始SOC为0.7')
plt.xlabel('时间')
plt.ylabel('蓄电池输出功率')
plt.title('不同初始SOC下动态规划求解的轨迹')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

data_1 = np.load('./data_master/output_dp_socs/soc_02_0.4.npy')
data_2 = np.load('./data_master/output_dp_socs/soc_02_0.5.npy')
data_3 = np.load('./data_master/output_dp_socs/soc_02_0.6.npy')
data_4 = np.load('./data_master/output_dp_socs/soc_02_0.7.npy')
plt.subplot(2,1,1)
plt.plot(data_1, 'b', label = '初始SOC为0.4')
plt.plot(data_2, 'r--', label = '初始SOC为0.5')
plt.plot(data_3, 'g-.', label = '初始SOC为0.6')
plt.plot(data_4, 'k', label = '初始SOC为0.7')
plt.xlabel('时间')
plt.ylabel('SOC')
plt.title('不同初始SOC下动态规划求解的轨迹')
plt.legend(bbox_to_anchor=(1, 0.48))
data_1 = np.load('./data_master/output_dp_socs/pbat_02_0.4.npy')
data_2 = np.load('./data_master/output_dp_socs/pbat_02_0.5.npy')
data_3 = np.load('./data_master/output_dp_socs/pbat_02_0.6.npy')
data_4 = np.load('./data_master/output_dp_socs/pbat_02_0.7.npy')
plt.subplot(2,1,2)
plt.plot(data_1, 'b', label = '初始SOC为0.4')
plt.plot(data_2, 'r--', label = '初始SOC为0.5')
plt.plot(data_3, 'g-.', label = '初始SOC为0.6')
plt.plot(data_4, 'k', label = '初始SOC为0.7')
plt.xlabel('时间')
plt.ylabel('蓄电池输出功率')
plt.title('不同初始SOC下动态规划求解的轨迹')
plt.legend(bbox_to_anchor=(1, 0.5))
plt.show()
'''

# ? dp输入
'''
data = np.load('./data_master/input_pfc_12.9/input_02_0.6.npy')
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(data[:,0],'b',label='pbus')
plt.title("pbus")
plt.subplot(3, 1, 2)
plt.plot(data[:,1],'k',label='soc')
plt.title("soc")
plt.subplot(3, 1, 3)
plt.plot(data[:,6],'r',label='pfc')
plt.title("pfc")
# plt.show()

data = np.load('./data_master/input_pfc_12.9/input_03_0.6.npy')
plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(data[:,0],'b',label='pbus')
plt.title("pbus")
plt.subplot(3, 1, 2)
plt.plot(data[:,1],'k',label='soc')
plt.title("soc")
plt.subplot(3, 1, 3)
plt.plot(data[:,6],'r',label='pfc')
plt.title("pfc")
# plt.show()

data = np.load('./data_master/input_pfc_12.9/input_04_0.6.npy')
plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(data[:,0],'b',label='pbus')
plt.title("pbus")
plt.subplot(3, 1, 2)
plt.plot(data[:,1],'k',label='soc')
plt.title("soc")
plt.subplot(3, 1, 3)
plt.plot(data[:,6],'r',label='pfc')
plt.title("pfc")
# plt.show()

data = np.load('./data_master/input_pfc_12.9/input_05_0.6.npy')
plt.figure(4)
plt.subplot(3, 1, 1)
plt.plot(data[:,0],'b',label='pbus')
plt.title("pbus")
plt.subplot(3, 1, 2)
plt.plot(data[:,1],'k',label='soc')
plt.title("soc")
plt.subplot(3, 1, 3)
plt.plot(data[:,6],'r',label='pfc')
plt.title("pfc")

data = np.load('./data_master/input_pfc_12.9/input_06_0.6.npy')
plt.figure(5)
plt.subplot(3, 1, 1)
plt.plot(data[:,0],'b',label='pbus')
plt.title("pbus")
plt.subplot(3, 1, 2)
plt.plot(data[:,1],'k',label='soc')
plt.title("soc")
plt.subplot(3, 1, 3)
plt.plot(data[:,6],'r',label='pfc')
plt.title("pfc")
plt.show()
'''

'''
# ? 绘制累积曲线
def drawCumulativeHist(heights):
    #创建累积曲线
    #第一个参数为待绘制的定量数据
    #第二个参数为划分的区间个数
    #normed参数为是否无量纲化
    #histtype参数为'step'，绘制阶梯状的曲线
    #cumulative参数为是否累积
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot((p_dp - p_lstm),'k')
    plt.xlabel('时间')
    plt.ylabel('误差')
    plt.title('神经网络与动态规划的误差分布图')
    plt.grid(True)
    plt.subplot(3,2,3)
    plt.hist(heights, 100, histtype='step', cumulative=False)
    plt.xlabel('误差')
    plt.ylabel('频次')
    plt.grid(True)
    plt.show()

# ? lstm vs dp
p_dp = np.load('./data_master/output_dp&lstm_12.9/pbat_dp.npy')
p_lstm = np.load('./data_master/output_dp&lstm_12.9/pbat_lstm.npy')

fc_dp = np.load('./data_master/output_dp&lstm_12.9/pfc_dp.npy')
fc_lstm = np.load('./data_master/output_dp&lstm_12.9/pfc_lstm.npy')

soc_dp = np.load('./data_master/output_dp&lstm_12.9/soc_dp.npy')
soc_lstm = np.load('./data_master/output_dp&lstm_12.9/soc_lstm.npy')

plt.figure(1)
plt.subplot(3,2,1)
plt.plot(p_dp,'r-.', label='动态规划')
plt.plot(p_lstm,'b', label='神经网络')
plt.ylabel('功率')
plt.legend()
plt.grid(True)
plt.title('验证集下的神经网络表现'+'\n'+'蓄电池输出功率')
plt.subplot(3,2,3)
plt.plot(fc_dp,'r-.', label='动态规划')
plt.plot(fc_lstm,'b', label='神经网络')
plt.ylabel('功率')
plt.legend()
plt.grid(True)
plt.title('燃料电池输出功率')
plt.subplot(3,2,5)
plt.plot(soc_dp,'r-.', label='动态规划')
plt.plot(soc_lstm,'b', label='神经网络')
plt.xlabel('时间')
plt.ylabel('SOC')
plt.title('SOC变化轨迹')
plt.legend()
plt.grid(True)
plt.show()

err = p_dp - p_lstm
np.save('err', err)
drawCumulativeHist(err)
'''
