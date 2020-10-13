import numpy as np
from matplotlib import pyplot as plt
import math
import random as rd


# * paramter define
samp_time = 1.0

# * about kinematic
veh_velc_1 = 20
veh_velc_2 = 23

veh_velc = (veh_velc_1+veh_velc_2) / 2.0
veh_acc = (veh_velc_2-veh_velc_1) / samp_time



def i_g_calc(veh_velc):

    if(veh_velc > 50.0/3.6):
        return 1.0
    elif(veh_velc > 20.0/3.6):
        return 1.71
    elif(veh_velc >10.0/3.6):
        return 3.09
    else:
        return 6.09

# * about transmission
i_fd = 1 # * final drive ratio
i_g = i_g_calc(veh_velc) # * transmission ratio

# * about mass
delt1 = 0.04
delt2 = 0.04
delt = 1 + delt1 + delt2 * i_g * i_g

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

# * about battery
SOC = 0.6
SOC_low_limit = 0.05
C_bat = 37.0 * 3600.0

ess_r_dis = 1.3*96.0/1000
ess_r_ch = 1.3 *96.0/1000

# * about fuel cell
fuel_consumption = 0.00
fuel_storage = 4000.0

fc_state_pre = 0.0
fc_state_cur = 0.0

fc_deg = 0.0

fc_pwr_low = 3.19e3
fc_pwr_high = 25.82e3

fc_pwr_pre = 0.0
fc_pwr_cur = 0.0

if SOC < SOC_low_limit and fuel_consumption < fuel_storage:
    print("NO POWER AVAILABLE")

    # todo break 跳出循环
    


F_roll_resis = f_roll_resis_coeff * veh_mass * veh_grav
F_ascend_resis = road_grade * veh_mass * veh_grav
F_air_resis = 0.5 * veh_CD * veh_A * air_density * veh_velc * veh_velc
F_acc_resis = delt * veh_mass * veh_acc

veh_force = F_roll_resis + F_ascend_resis + F_air_resis + F_acc_resis

# * about wheel
wheel_torque = veh_force * wheel_radius
wheel_speed = veh_velc * wheel_radius

# * about final drive
fd_torque = wheel_torque / i_fd
fd_speed = wheel_speed * i_fd

# * about gearbox
gb_torque = fd_torque / i_g
gb_speed = fd_speed * i_g


# * about electric motor
em_eff_tab=np.array([[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7],
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


em_torq_map=np.array([0,27.1137,54.2274,81.3411,108.4547,135.5684,162.6821,189.7958,216.9095,244.0232,271.1368])
em_spd_map=np.array([ 0,104.7,209.4,314.2,418.9,523.6,628.3,733.0,837.8,942.5,1047.2])


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

# * bus power

p_bus = gb_speed * gb_torque/em_eff



# todo energy manegement strategy

p_bat = rd.randint(-30,30)*1000

p_fc  = p_bus - p_bat


print(p_bus)
print(p_bat)
print(p_fc)

# todo *************************end


# * fc start detection

if(fuel_consumption<fuel_storage and p_fc>0):
    fc_state_pre = fc_state_cur
    fc_state_cur = 1.0
else:
    fc_state_pre = fc_state_cur
    fc_state_cur = 0.0
    

fc_pwr_map = np.array([0,4800,9500,14100,18500,22700,26800,29700,32500])

# * power-efficiency
# ? fc_eff

fc_eff_map = np.array([0.00,0.55,0.525,0.515,0.500,0.495,0.49,0.48,0.47])

for i in range(len(fc_pwr_map)):
    if(p_fc>fc_pwr_map[i]):
        continue
    else:
        break

fc_eff = (p_fc-fc_pwr_map[i-1]) * (fc_eff_map[i]-fc_eff_map[i-1])/ (fc_pwr_map[i]-fc_pwr_map[i-1])+fc_eff_map[i-1]

# * fuel consumption
# ? fuel_consumption
fc_fuel_map = np.array([0.0000001,0.0733,0.151,0.2307,0.3074,0.3812,0.4577,0.5166,0.5758])

for i in range(len(fc_pwr_map)):
    if(p_fc > fc_pwr_map[i]):
        continue
    else:
        break

fc_fuel_cons = ((p_fc-fc_pwr_map[i-1]) * (fc_fuel_map[i]-fc_fuel_map[i-1])/ (fc_pwr_map[i]-fc_pwr_map[i-1])+fc_fuel_map[i-1])*samp_time

fuel_consumption = fuel_consumption + fc_fuel_cons

# * fc durability
# ? fc_deg

if(fc_state_cur == 1 and fc_state_pre == 0):
    fc_deg = fc_deg + 1.96e-3

if(p_fc<fc_pwr_low and fc_state_cur==1):
    fc_deg = fc_deg + 1.26e-3*samp_time/3600

fc_pwr_pre = fc_pwr_cur
fc_pwr_cur = p_fc

fc_deg = fc_deg + 5.93e-5*abs(fc_pwr_pre-fc_pwr_cur)/(2*(fc_pwr_high-fc_pwr_low))

if(p_fc>fc_pwr_high):
    fc_deg = fc_deg + 1.47e-3*samp_time/3600


# * battery model
# ? SOC

ess_voc_map = np.array([3.438*96,3.533*96,3.597*96,3.627*96,3.658*96,3.719*96,3.785*96,3.858*96 ,3.946*96,4.049*96,4.18*96])
ess_soc_map = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

for i in range(len(ess_soc_map)):
    if(SOC > ess_soc_map[i]):
        continue
    else:
        break

ess_voc = ((SOC-ess_soc_map[i-1]) * (ess_voc_map[i]-ess_voc_map[i-1])/ (ess_soc_map[i]-ess_soc_map[i-1])+ess_voc_map[i-1])

if(p_bat>0):
    ess_state = "charge"
else:
    ess_state = "discharge"

if(ess_state == "charge"):
    SOC = SOC - ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_ch*abs(p_bat)))/2.0*ess_r_ch)*samp_time/C_bat

if(ess_state == "discharge"):
    SOC = SOC + ((ess_voc-math.sqrt(ess_voc**2-4.0*ess_r_dis*abs(p_bat)))/2.0*ess_r_dis)*samp_time/C_bat


print(fc_state_cur)
print(fc_state_pre)

print(fc_pwr_pre)
print(fc_pwr_cur)
print('\n')

print(SOC)
print(fc_deg)
print(fc_eff)
print(fuel_consumption)