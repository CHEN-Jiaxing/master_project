#* **********************
#* Author: CHEN, Jiaxing
#* Function: according to the pbus and pfc 
#*           calculate the fuel consumption and fc degration
#* Date: 2021.01.09
#* Modified by:
#* Changes:
#* **********************

import numpy as np
from matplotlib import pyplot as plt
import math
import random as rd
import matplotlib
matplotlib.rc("font",family='DengXian')
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter

# * veh_mass
veh_mass = 1670.0 # *Kg
veh_grav = 9.8 # * m/(s^2)

# * about aerodynamic
veh_CD = 0.335
veh_A = 2.0

# * about road condition
road_grade = 0.0 
air_density = 1.2

# * about wheel
f_roll_resis_coeff = 0.015 
wheel_radius = 0.2820

# * final_drive
i_fd = 1

# * delt coefficient of mass
delt1 = 0.04
delt2 = 0.04

# * about battery
SOC_STOP = 0.05
SOC_low_limit = 0.4
SOC_high_limit = 0.8
C_bat = 37.0 * 3600.0

ess_r_dis = 1.3 * 96.0 / 1000
ess_r_ch = 1.3 * 96.0 / 1000

ess_pwr_ch = 19250
ess_pwr_dis = 66000

ess_voc_map = np.array([3.438*96,3.533*96,3.597*96,3.627*96,3.658*96,3.719*96,3.785*96,3.858*96 ,3.946*96,4.049*96,4.18*96])
ess_soc_map = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# * about fuel cell
fuel_storage = 4000.0
# * 双堆燃料电池
fc_pwr_low = 3.19e3 * 2
fc_pwr_high = 25.82e3 * 2

# * 单堆特性
fc_pwr_map = np.array([0,4800,9500,14100,18500,22700,26800,29700,32500])
fc_eff_map = np.array([0.00,0.55,0.525,0.515,0.500,0.495,0.49,0.48,0.47])
fc_fuel_map = np.array([0.0000001,0.0733,0.151,0.2307,0.3074,0.3812,0.4577,0.5166,0.5758])

# * about electric motor
em_eff_tab = np.array([[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],
                    [0.7,0.77,0.81,0.82,0.82,0.82,0.81,0.8,0.79,0.78,0.78],
                    [0.7,0.82,0.85,0.86,0.87,0.88,0.87,0.86,0.86,0.86,0.85],
                    [0.7,0.87,0.89,0.9,0.9,0.9,0.9,0.89,0.88,0.87,0.86],
                    [0.7,0.88,0.91,0.91,0.91,0.9,0.88,0.87,0.85,0.82,0.81],
                    [0.7,0.89,0.91,0.91,0.9,0.87,0.85,0.82,0.82,0.82,0.82],
                    [0.7,0.9,0.91,0.9,0.86,0.82,0.79,0.78,0.79,0.79,0.79],
                    [0.7,0.91,0.91,0.88,0.8,0.78,0.78,0.78,0.78,0.78,0.78],
                    [0.70,0.92,0.90,0.80,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
                    [0.70,0.92,0.88,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
                    [0.70,0.92,0.80,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78]])

em_torq_map = np.array([0,27.1137,54.2274,81.3411,108.4547,135.5684,162.6821,189.7958,216.9095,244.0232,271.1368])
em_spd_map = np.array([ 0,104.7,209.4,314.2,418.9,523.6,628.3,733.0,837.8,942.5,1047.2])


# * gear ratio calc
def i_g_calc(veh_velc):
    '''
        if(veh_velc > 50.0/3.6):
            return 1.0
        elif(veh_velc > 20.0/3.6):
            return 1.71
        elif(veh_velc >10.0/3.6):
            return 3.09
        else:
            return 6.09
    '''
    return 1.0
    
def veh_force_cal(f_roll_resis_coeff,veh_mass,veh_grav,road_grade,veh_CD,veh_A,air_density,veh_velc,delt,veh_acc):
    F_roll_resis = f_roll_resis_coeff * veh_mass * veh_grav
    F_ascend_resis = road_grade * veh_mass * veh_grav
    F_air_resis = 0.5 * veh_CD * veh_A * air_density * veh_velc ** 2
    F_acc_resis = delt * veh_mass * veh_acc

    return  F_roll_resis + F_ascend_resis + F_air_resis + F_acc_resis

# * about electric motor
def em_eff_cal(gb_torque,gb_speed):
    for i in range(len(em_torq_map)):
        if(gb_torque>em_torq_map[i]):
            continue
        else:
            break
    for j in range(len(em_spd_map)):
            if(gb_speed>em_spd_map[j]):
                continue
            else:
                break
        
    em_eff = (em_eff_tab[j-1][i-1] + em_eff_tab[j][i-1] + em_eff_tab[j-1][i] + em_eff_tab[j][i])/4
    return em_eff

# * power-efficiency one stack

def fc_eff_cal(p_fc):
    for i in range(len(fc_pwr_map)):
        if(p_fc>fc_pwr_map[i]):
            continue
        else:
            break

    fc_eff = (p_fc-fc_pwr_map[i-1]) * (fc_eff_map[i]-fc_eff_map[i-1])/ (fc_pwr_map[i]-fc_pwr_map[i-1])+fc_eff_map[i-1]
    return fc_eff

# * fuel consumption one stack
def fuel_cons_cal(p_fc):
    for i in range(len(fc_pwr_map)):
        if(p_fc > fc_pwr_map[i]):
            continue
        else:
            break

    fc_fuel_cons = ((p_fc-fc_pwr_map[i-1]) * (fc_fuel_map[i]-fc_fuel_map[i-1])/ (fc_pwr_map[i]-fc_pwr_map[i-1])+fc_fuel_map[i-1])*samp_time
    return fc_fuel_cons

# * fc_deg one stack
def fc_deg_cal(fc_state_cur,fc_state_pre,fc_deg,fc_pwr_pre,fc_pwr_cur):
    if(fc_state_cur == 1 and fc_state_pre == 0):
        fc_deg = fc_deg + 1.96e-3

    if(fc_pwr_cur<fc_pwr_low and fc_state_cur==1):
        fc_deg = fc_deg + 1.26e-3*samp_time/3600

    fc_deg = fc_deg + 5.93e-5*abs(fc_pwr_pre-fc_pwr_cur)/(2*(fc_pwr_high-fc_pwr_low))

    if(fc_pwr_cur>fc_pwr_high):
        fc_deg = fc_deg + 1.47e-3*samp_time/3600

    return fc_deg

# * battery model
# * SOC
def SOC_cal(p_bat,SOC):
    for i in range(len(ess_soc_map)):
        if(SOC > ess_soc_map[i]):
            continue
        else:
            break

    ess_voc = ((SOC-ess_soc_map[i-1]) * (ess_voc_map[i]-ess_voc_map[i-1])/ (ess_soc_map[i]-ess_soc_map[i-1])+ess_voc_map[i-1])
    if(p_bat>0):
        SOC = SOC - ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_ch*abs(p_bat)))/(2.0*ess_r_ch))*samp_time/C_bat
    else:
        SOC = SOC + ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_dis*abs(p_bat)))/(2.0*ess_r_dis))*samp_time/C_bat

    return SOC

# ! input is p_bus and p_fc
data = np.load('./data_master/input_pfc_12.9/input_pfc_vld.npy')
data = np.concatenate((data,data,data,data,data),axis=0)
p_bus_list = savgol_filter(data[:,0], 31, 3, mode= 'nearest')* 1000
p_bus_list = p_bus_list[0:9135]
p_bat_list = np.load('./p_ann.npy')* 1000
print(len(p_bat_list))
# p_bus_list = np.load("./results/p_bus.npy") * 1000
# p_fc_list = np.load("./results/p_dp.npy") * 1000

samp_time = 1.0
t_list = np.arange(len(p_bus_list)) * samp_time

# ! 根据当前值与前一步值计算下一步值，导致矩阵维度的不一致
# * the first element presents the initialized condition
SOC_list = np.array([0.6])
fuel_consumption_list = np.array([0.0])
fc_state_list = np.array([0])
fc_deg_list = np.array([0.0])
fc_eff_list = np.array([0.0])
# p_bus_list = np.array([0.0])
# p_bat_list = np.array([0.0])
# p_fc_list = np.array([0.0])
p_fc_list = np.array([0.0])

for i in range(len(p_bus_list)):
    SOC = SOC_list[i]
    fuel_consumption = fuel_consumption_list[i]
    p_bus = p_bus_list[i]
    # p_fc = p_fc_list[i]
    p_bat = p_bat_list[i]

    if SOC >= SOC_high_limit:
        if p_bus >= ess_pwr_dis:
            p_bat_list[i] = ess_pwr_dis
        else:
            p_bat_list[i] = p_bus
    p_bat = p_bat_list[i]
    p_fc = p_bus - p_bat
    p_fc_list = np.append(p_fc_list, p_fc)
    
    # * fc start detection
    if (fuel_consumption < fuel_storage and p_fc > 0):
        fc_state_list = np.append(fc_state_list, 1.0)
    else:
        fc_state_list = np.append(fc_state_list, 0.0)
    
    fc_state_pre = fc_state_list[i]
    fc_state_cur = fc_state_list[i+1]

    fc_eff = fc_eff_cal(p_fc / 2.0)
    fc_eff_list = np.append(fc_eff_list, fc_eff)

    fc_fuel_cons = fuel_cons_cal(p_fc / 2.0) * 2
    fuel_consumption = fuel_consumption + fc_fuel_cons
    fuel_consumption_list = np.append(fuel_consumption_list, fuel_consumption)

    fc_pwr_pre = p_fc_list[i]
    fc_pwr_cur = p_fc

    fc_deg = fc_deg_list[i]
    fc_deg = fc_deg + 2 * fc_deg_cal(fc_state_cur,fc_state_pre,0.0,fc_pwr_pre/2.0,fc_pwr_cur/2.0)
    fc_deg_list = np.append(fc_deg_list, fc_deg)

    '''
    p_bat = p_bus - p_fc
    p_bat_list = np.append(p_bat_list, p_bat)
    '''

    SOC = SOC_cal(p_bat,SOC)
    SOC_list = np.append(SOC_list, SOC)

t_list = np.append(t_list, t_list[-1]+samp_time)

plt.figure(1)
plt.subplot(2,2,1)
plt.plot(t_list,p_fc_list,'b')
plt.xlabel('时间')
plt.ylabel('功率')
plt.title('燃料电池输出功率')
plt.subplot(2,2,2)
plt.plot(t_list,SOC_list,'b')
plt.xlabel('时间')
plt.ylabel('SOC')
plt.title('电量水平变化')
plt.subplot(2,2,3)
plt.plot(t_list,fuel_consumption_list,'b')
plt.xlabel('时间')
plt.ylabel('氢耗量')
plt.title('氢耗量量变化')
plt.subplot(2,2,4)
plt.plot(t_list,fc_deg_list,'b')
plt.xlabel('时间')
plt.ylabel('衰减率')
plt.title('燃料电池衰减率变化')
plt.show()

print(fuel_consumption_list[-1])
print(fc_deg_list[-1])
print(SOC_list[-1])

np.save("./fcon_ann.npy", fuel_consumption_list)
np.save("./deg_ann.npy", fc_deg_list)
np.save("./eff_ann.npy", fc_eff_list)
np.save("soc_ann", SOC_list)
np.save("pfc_ann", p_fc_list)
np.save("pbat_ann", p_bat_list)

print("=====END=====")