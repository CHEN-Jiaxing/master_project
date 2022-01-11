#* **********************
#* Author: CHEN, Jiaxing
#* Function: calculate the optimal power split under certain drive conditon
#* Date: 2021.01.15
#* Modified by:
#* Changes:
#* **********************

import matplotlib
matplotlib.rc("font",family='DengXian')
import numpy as np
from matplotlib import pyplot as plt
import math
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

# * about soc
# ! expected SOC and initialized SOCs
SOC_exp = 0.6
SOC_init = 0.6
SOC_low_limit = 0.4
SOC_high_limit = 0.8

# * about battery
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
    return 1.0

def veh_force_cal(f_roll_resis_coeff,veh_mass,veh_grav,road_grade,veh_CD,veh_A,air_density,veh_velc,delt,veh_acc):
    F_roll_resis = f_roll_resis_coeff * veh_mass * veh_grav
    F_ascend_resis = road_grade * veh_mass * veh_grav
    F_air_resis = 0.5 * veh_CD * veh_A * air_density * (veh_velc ** 2)
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
def VOC_cal(SOC):
    for i in range(len(ess_soc_map)):
        if(SOC > ess_soc_map[i]):
            continue
        else:
            break

    ess_voc = ((SOC-ess_soc_map[i-1]) * (ess_voc_map[i]-ess_voc_map[i-1])/ (ess_soc_map[i]-ess_soc_map[i-1])+ess_voc_map[i-1])
    return ess_voc

def SOC_cal(p_bat,SOC):
    ess_voc = VOC_cal(SOC)
    if(p_bat>0):
        SOC = SOC - ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_ch*abs(p_bat)))/2.0*ess_r_ch)*samp_time/C_bat
    else:
        SOC = SOC + ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_dis*abs(p_bat)))/2.0*ess_r_dis)*samp_time/C_bat

    return SOC
fileno_list = [2,3,4,5,6]
for fileno in fileno_list:
    # * about kinematic m*s**(-1)    m*s**(-2)
    # ! !!!input and output parameters setting
    # fileno = int(input("输入文件编号"))
    data = np.load('./data_master/input_dp/input_dp_' + str(fileno) + '.npy')
    data = np.concatenate((data,data),axis=0)
    # data = np.concatenate((data,data,data,data,data),axis=0)
    veh_velc_list = data[:,6]
    veh_acc_list = data[:,7]
    veh_velc_list= savgol_filter(veh_velc_list, 51, 3, mode= 'nearest')
    veh_acc_list= savgol_filter(veh_acc_list, 51, 3, mode= 'nearest')
    samp_time = 1.0
    t_list = np.arange(len(veh_velc_list)) * samp_time
    N = len(t_list)
    p_bus_list = np.array([])

    # * Pbus calculation
    for i in range(len(veh_velc_list)):
        veh_velc = veh_velc_list[i]
        veh_acc = veh_acc_list[i]
        
        # * transmission ratio
        i_g = i_g_calc(veh_velc) 

        # * 转换质量参数计算
        delt = 1 + delt1 + delt2 * i_g ** 2 
        veh_force = veh_force_cal(f_roll_resis_coeff,veh_mass,veh_grav,road_grade,veh_CD,veh_A,air_density,veh_velc,delt,veh_acc)

        # * about wheel
        wheel_torque = veh_force * wheel_radius
        wheel_speed = veh_velc * wheel_radius

        # * about final drive
        fd_torque = wheel_torque / i_fd
        fd_speed = wheel_speed * i_fd

        # * about gearbox
        gb_torque = fd_torque / i_g
        gb_speed = fd_speed * i_g

        # * bus power
        em_eff = em_eff_cal(gb_torque, gb_speed)
        p_bus = gb_speed * gb_torque * 2.5 / em_eff
        p_bus_list = np.append(p_bus_list, p_bus)
    '''
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(veh_velc_list,'b',label="速度")
    plt.plot(veh_acc_list,'r',label="加速度")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(p_bus_list,'b',label="总线功率需求")
    plt.xlabel('时间(s)', fontsize = 12)
    plt.legend()
    plt.show()
    '''
    '''
    print(len(veh_velc_list))
    print(len(veh_acc_list))
    print(len(t_list))
    print(len(p_bus_list))
    print("----------")
    '''

    # ! cost function
    def c2g_cal(fc_pwr_grid, fc_pwr_pre, SOC_cur, SOC_exp, SOC_pre):
        c2g = np.zeros(len(fc_pwr_grid))
        for i in range(len(fc_pwr_grid)):
            if(fc_pwr_grid[i] > 0):
                fc_state_cur = 1
            else:
                fc_state_cur = 0

            if(fc_pwr_pre > 0):
                fc_state_pre = 1
            else:
                fc_state_pre = 0
            c2g[i] = 79800 * (fc_deg_cal(fc_state_cur, fc_state_pre, 0, fc_pwr_pre, fc_pwr_grid[i])/10)+\
                    75e2 * (SOC_cur - SOC_exp)**2 + 70 * (fuel_cons_cal(fc_pwr_grid[i]) + 240 * 13 * (SOC_pre - SOC_cur))/1000
        return c2g

    # * value calculation
    def V_nxt_i_cal(SOC_next_i, SOC_grid, Value_cut):
        for i in range(len(SOC_grid)):
            if(SOC_next_i > SOC_grid[i]):
                continue
            else:
                break

        V_nxt_i = (SOC_next_i-SOC_grid[i-1]) * (Value_cut[i]-Value_cut[i-1])/ (SOC_grid[i]-SOC_grid[i-1]) + Value_cut[i-1]
        return V_nxt_i

    def V_nxt_cal(SOC_next, SOC_grid, Value_cut):
        V_nxt = np.zeros(len(SOC_next))
        for i in range(len(SOC_next)):
            V_nxt[i] = V_nxt_i_cal(SOC_next[i], SOC_grid, Value_cut)
        return V_nxt

    # soc 等分数量
    soc_num = 50
    soc_step = (SOC_high_limit - SOC_low_limit) / soc_num
    SOC_grid = np.arange(SOC_low_limit, SOC_high_limit + soc_step, soc_step, float).reshape(soc_num + 1,1)
    n_soc = len(SOC_grid)

    # 损失值
    Value = np.zeros((n_soc, N))

    # 电池理想输出表
    ess_pwr_opt = np.zeros((n_soc, N - 1))

    # 记录前一时刻Pfc， 以便计算衰减
    fc_pwr_opt = np.zeros((n_soc, N))

    # * 上一时刻SOC
    SOC_pre = np.ones(len(SOC_grid)) * SOC_init

    # * calculate the optimal power table under different soc and time
    for i in range(N-2, -1, -1):
        for j in range(n_soc):
            ess_pwr_lb = max(((SOC_high_limit - SOC_grid[j])*C_bat*VOC_cal(SOC_grid[j])/-samp_time), -ess_pwr_ch, p_bus_list[i] - fc_pwr_high)
            ess_pwr_ub = min(((SOC_low_limit - SOC_grid[j])*C_bat*VOC_cal(SOC_grid[j])/-samp_time), ess_pwr_dis, p_bus_list[i])
            
            # 划分蓄电池区间
            ess_pwr_grid = np.linspace(ess_pwr_lb, ess_pwr_ub, 100)
            fc_pwr_grid = p_bus_list[i] - ess_pwr_grid
            
            # 计算单步损失
            c2g = c2g_cal(fc_pwr_grid, fc_pwr_opt[j][i+1], SOC_grid[j], SOC_exp, SOC_pre[j])
            SOC_next = SOC_grid[j] - (samp_time * ess_pwr_grid / (C_bat * VOC_cal(SOC_grid[j])))
            
            # 估算损失
            V_nxt = V_nxt_cal(SOC_next, SOC_grid, Value[:,i+1].reshape(soc_num + 1,1))
            k = np.argmin(c2g + V_nxt)
            Value[j,i] = c2g[k] + V_nxt[k]
            ess_pwr_opt[j][i] = ess_pwr_grid[k]
            SOC_pre[j] = SOC_next[k]
            fc_pwr_opt[j][i] = p_bus_list[i] - ess_pwr_grid[k]
        print("===", i, "===")

    # *     
    def ess_pwr_act_cal(SOC_act_i, SOC_grid, ess_pwr_opt):
        for i in range(len(SOC_grid)):
            if(SOC_act_i > SOC_grid[i]):
                continue
            else:
                break

        ess_pwr_act = (SOC_act_i-SOC_grid[i-1]) * (ess_pwr_opt[i]-ess_pwr_opt[i-1])/ (SOC_grid[i]-SOC_grid[i-1])+ess_pwr_opt[i-1]
        return ess_pwr_act

    # ! real start function
    def run(SOC_init, N, SOC_grid, ess_pwr_opt, p_bus_list):
        SOC_act = np.zeros(N)
        SOC_act[0] = SOC_init
        ess_pwr_act = np.zeros(N)
        fc_pwr_act = np.zeros(N)
        samp_time = 1
        C_bat = 37.0 * 3600.0

        for i in range(N-1):
            ess_pwr_act[i] = ess_pwr_act_cal(SOC_act[i], SOC_grid, ess_pwr_opt[:,i])
            fc_pwr_act[i] = p_bus_list[i] - ess_pwr_act[i]
            
            if fc_pwr_act[i] < 0.0:
                fc_pwr_act[i] = 0.0
                ess_pwr_act[i] = p_bus_list[i]
            
            SOC_act[i+1] = SOC_act[i] - ((samp_time * ess_pwr_act[i])/(C_bat * VOC_cal(SOC_act[i])))
            
        return [ess_pwr_act, fc_pwr_act, SOC_act]

    # ! /ess_pwr_act：蓄电池输出功率 /fc_pwr_act：fc输出功率 /SOC：蓄电SOC 
    [ess_pwr_act, fc_pwr_act, SOC]= run(SOC_init,N,SOC_grid,ess_pwr_opt,p_bus_list)
    '''
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(ess_pwr_act, 'r',label='蓄电池')
    plt.xlabel("时间")
    plt.ylabel("输出功率")
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(fc_pwr_act, 'g',label='燃料电池')
    plt.xlabel("时间")
    plt.ylabel("输出功率")
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(SOC, 'k',label='初始时刻SOC为'+str(SOC_init))
    plt.xlabel("时间")
    plt.ylabel("SOC")
    # plt.ylim((0.69,0.71))
    plt.grid(True)
    plt.legend()
    plt.show()
    '''
    f_fc_name = "pfc_0" + str(fileno) + "_"+ str(SOC_init)+".npy"
    f_bat_name = "pbat_0" + str(fileno) + "_"+ str(SOC_init)+".npy"
    f_soc_name = "soc_0" + str(fileno) + "_"+ str(SOC_init)+".npy"
    np.save(f_fc_name,fc_pwr_act)
    np.save(f_bat_name,ess_pwr_act)
    np.save(f_soc_name,SOC)

    '''
    print(len(p_bus_list))
    print(len(ess_pwr_act))
    print(len(fc_pwr_act))
    print(len(SOC))
    print(len(veh_velc_list))
    print(len(veh_acc_list))
    print("----------")
    '''

    p_bus_list = p_bus_list.reshape(len(p_bus_list),1) / 1000
    ess_pwr_act = ess_pwr_act.reshape(len(ess_pwr_act), 1) / 1000
    SOC = SOC.reshape(len(SOC),1) * C_bat / 1000
    SOC_low_list = np.ones(len(SOC)).reshape(len(SOC),1) * SOC_low_limit * C_bat / 1000
    SOC_high_list = np.ones(len(SOC)).reshape(len(SOC),1) * SOC_high_limit * C_bat / 1000
    veh_velc_list = veh_velc_list.reshape(len(veh_velc_list),1)
    veh_acc_list = veh_acc_list.reshape(len(veh_acc_list),1)
    fc_pwr_act = fc_pwr_act.reshape(len(fc_pwr_act),1) / 1000

    chekuang_input_neur = np.hstack((p_bus_list, SOC, SOC_low_list, SOC_high_list,
    veh_velc_list, veh_acc_list, ess_pwr_act))

    # ! save data
    # filename = "input_08_" + str(SOC_init) + "-" + str(SOC_low_limit) + "-" + str(SOC_high_limit)
    filename = "input_0" + str(fileno) + "_" + str(SOC_init) + ".npy"
    np.save(filename, chekuang_input_neur)
    print("=====END=====")
