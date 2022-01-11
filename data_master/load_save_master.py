#* **********************
#* Author: CHEN, Jiaxing
#* Function: 数据转存
#* Date: 2021.03.01
#* Modified by:
#* Changes:
#* **********************

import matplotlib
matplotlib.rc("font",family='DengXian')
import numpy as np
from matplotlib import pyplot as plt
import subsidiary as sub

'''
# ? input for velocity and acceleration
data_0 = np.load('./input_va_master/input_1_001_14.npy')
data_1 = np.load('./input_va_master/input_1_001_27.npy')
data_2 = np.load('./input_va_master/input_1_001_42.npy')
data_3 = np.load('./input_va_master/input_1_001_7.npy')
data_4 = np.load('./input_va_master/input_1_002_14.npy')
data_5 = np.load('./input_va_master/input_1_002_8.npy')
data_6 = np.load('./input_va_master/input_1_003_11.npy')
data_7 = np.load('./input_va_master/input_1_003_20.npy')
data_8 = np.load('./input_va_master/input_1_003_35.npy')
data_9 = np.load('./input_va_master/input_1_003_45.npy')
data_10 = np.load('./input_va_master/input_1_004_10.npy')
data_11 = np.load('./input_va_master/input_1_004_25.npy')
data_12 = np.load('./input_va_master/input_1_004_7.npy')
data_13 = np.load('./input_va_master/input_1_005_20.npy')
data_14 = np.load('./input_va_master/input_1_005_29.npy')
data_15 = np.load('./input_va_master/input_1_005_5.npy')
data_16 = np.load('./input_va_master/input_1_005_9.npy')
data_17 = np.load('./input_va_master/input_1_006_11.npy')
data_18 = np.load('./input_va_master/input_1_006_33.npy')
data_19 = np.load('./input_va_master/input_1_006_5.npy')
data_20 = np.load('./input_va_master/input_1_007_36.npy')
data_21 = np.load('./input_va_master/input_1_007_8.npy')
data_22 = np.load('./input_va_master/input_1_008_17.npy')
data_23 = np.load('./input_va_master/input_1_008_22.npy')
data_24 = np.load('./input_va_master/input_1_008_32.npy')
data_25 = np.load('./input_va_master/input_1_009_17.npy')
data_26 = np.load('./input_va_master/input_1_009_27.npy')
data_27 = np.load('./input_va_master/input_1_010_14.npy')
data_28 = np.load('./input_va_master/input_1_010_20.npy')
data_29 = np.load('./input_va_master/input_1_010_29.npy')
data_30 = np.load('./input_va_master/input_1_010_36.npy')
data_31 = np.load('./input_va_master/input_1_011_13.npy')
data_32 = np.load('./input_va_master/input_1_011_24.npy')
data_33 = np.load('./input_va_master/input_1_011_30.npy')
data_34 = np.load('./input_va_master/input_1_011_8.npy')
data_35 = np.load('./input_va_master/input_1_012_13.npy')
data_36 = np.load('./input_va_master/input_1_012_34.npy')
data_37 = np.load('./input_va_master/input_1_012_9.npy')
data_38 = np.load('./input_va_master/input_1_013_13.npy')
data_39 = np.load('./input_va_master/input_1_013_18.npy')
data_40 = np.load('./input_va_master/input_1_013_19.npy')
data_41 = np.load('./input_va_master/input_1_013_21.npy')
data_42 = np.load('./input_va_master/input_1_013_27.npy')
data_43 = np.load('./input_va_master/input_1_013_37.npy')
data_44 = np.load('./input_va_master/input_1_013_4.npy')
data_45 = np.load('./input_va_master/input_1_014_15.npy')
data_46 = np.load('./input_va_master/input_1_014_30.npy')
data_47 = np.load('./input_va_master/input_1_014_5.npy')
data_48 = np.load('./input_va_master/input_1_015_7.npy')
data_49 = np.load('./input_va_master/input_1_016_7.npy')
data_50 = np.load('./input_va_master/input_2_001_14.npy')
data_51 = np.load('./input_va_master/input_2_001_6.npy')
data_52 = np.load('./input_va_master/input_2_001_8.npy')
data_53 = np.load('./input_va_master/input_2_002_18.npy')
data_54 = np.load('./input_va_master/input_2_002_25.npy')
data_55 = np.load('./input_va_master/input_2_002_35.npy')
data_56 = np.load('./input_va_master/input_2_002_5.npy')
data_57 = np.load('./input_va_master/input_2_003_20.npy')
data_58 = np.load('./input_va_master/input_2_003_32.npy')
data_59 = np.load('./input_va_master/input_2_003_5.npy')
data_60 = np.load('./input_va_master/input_2_003_7.npy')

data_0 = sub.time_shift(data_0, 60)
data_1 = sub.time_shift(data_1, 60)
data_2 = sub.time_shift(data_2, 60)
data_3 = sub.time_shift(data_3, 60)
data_4 = sub.time_shift(data_4, 60)
data_5 = sub.time_shift(data_5, 60)
data_6 = sub.time_shift(data_6, 60)
data_7 = sub.time_shift(data_7, 60)
data_8 = sub.time_shift(data_8, 60)
data_9 = sub.time_shift(data_9, 60)
data_10 = sub.time_shift(data_10, 60)
data_11 = sub.time_shift(data_11, 60)
data_12 = sub.time_shift(data_12, 60)
data_13 = sub.time_shift(data_13, 60)
data_14 = sub.time_shift(data_14, 60)
data_15 = sub.time_shift(data_15, 60)
data_16 = sub.time_shift(data_16, 60)
data_17 = sub.time_shift(data_17, 60)
data_18 = sub.time_shift(data_18, 60)
data_19 = sub.time_shift(data_19, 60)
data_20 = sub.time_shift(data_20, 60)
data_21 = sub.time_shift(data_21, 60)
data_22 = sub.time_shift(data_22, 60)
data_23 = sub.time_shift(data_23, 60)
data_24 = sub.time_shift(data_24, 60)
data_25 = sub.time_shift(data_25, 60)
data_26 = sub.time_shift(data_26, 60)
data_27 = sub.time_shift(data_27, 60)
data_28 = sub.time_shift(data_28, 60)
data_29 = sub.time_shift(data_29, 60)
data_30 = sub.time_shift(data_30, 60)
data_31 = sub.time_shift(data_31, 60)
data_32 = sub.time_shift(data_32, 60)
data_33 = sub.time_shift(data_33, 60)
data_34 = sub.time_shift(data_34, 60)
data_35 = sub.time_shift(data_35, 60)
data_36 = sub.time_shift(data_36, 60)
data_37 = sub.time_shift(data_37, 60)
data_38 = sub.time_shift(data_38, 60)
data_39 = sub.time_shift(data_39, 60)
data_40 = sub.time_shift(data_40, 60)
data_41 = sub.time_shift(data_41, 60)
data_42 = sub.time_shift(data_42, 60)
data_43 = sub.time_shift(data_43, 60)
data_44 = sub.time_shift(data_44, 60)
data_45 = sub.time_shift(data_45, 60)
data_46 = sub.time_shift(data_46, 60)
data_47 = sub.time_shift(data_47, 60)
data_48 = sub.time_shift(data_48, 60)
data_49 = sub.time_shift(data_49, 60)
data_50 = sub.time_shift(data_50, 60)
data_51 = sub.time_shift(data_51, 60)
data_52 = sub.time_shift(data_52, 60)
data_53 = sub.time_shift(data_53, 60)
data_54 = sub.time_shift(data_54, 60)
data_55 = sub.time_shift(data_55, 60)
data_56 = sub.time_shift(data_56, 60)
data_57 = sub.time_shift(data_57, 60)
data_58 = sub.time_shift(data_58, 60)
data_59 = sub.time_shift(data_59, 60)
data_60 = sub.time_shift(data_60, 60)

# * prediction
data_va_p = np.concatenate((
data_0,data_1,data_2,data_3,data_4,data_6,data_7,data_8,data_10,
data_11,data_12,data_13,data_14,data_16,data_17,data_18,data_20,
data_21,data_22,data_23,data_24,data_26,data_27,data_28,data_30,
data_31,data_32,data_33,data_34,data_36,data_37,data_38,data_40,
data_41,data_42,data_43,data_44,data_46,data_47,data_48
),axis=0)
# * input_预测时间_va_prediction
np.save("input_60_va_pre", data_va_p)
print("===========End==========")

# * validation
data_va_1_v = np.concatenate((
data_5,data_9,
data_15,data_19,
data_25,data_29,
data_35,data_39,
data_45,data_49
),axis=0)
# * input_车道数_预测时间_va_validation
np.save("input_1_60_va_vld", data_va_1_v)

data_va_2_v = np.concatenate((
data_51,data_52,data_53,data_54,data_55,data_56,data_57,data_58,data_59,data_50
),axis=0)
# * input_车道数_预测时间_va_validation
np.save("input_2_60_va_vld", data_va_2_v)
print("===========End==========")
'''

'''
# ? input for dp
data_0 = np.load('./data_master/input_va_origin/input_1_001_14.npy')
data_1 = np.load('./data_master/input_va_origin/input_1_001_27.npy')
data_2 = np.load('./data_master/input_va_origin/input_1_001_42.npy')
data_3 = np.load('./data_master/input_va_origin/input_1_001_7.npy')
data_4 = np.load('./data_master/input_va_origin/input_1_002_14.npy')
data_5 = np.load('./data_master/input_va_origin/input_1_002_8.npy')
data_6 = np.load('./data_master/input_va_origin/input_1_003_11.npy')
data_7 = np.load('./data_master/input_va_origin/input_1_003_20.npy')
data_8 = np.load('./data_master/input_va_origin/input_1_003_35.npy')
data_9 = np.load('./data_master/input_va_origin/input_1_003_45.npy')
data_10 = np.load('./data_master/input_va_origin/input_1_004_10.npy')
data_11 = np.load('./data_master/input_va_origin/input_1_004_25.npy')
data_12 = np.load('./data_master/input_va_origin/input_1_004_7.npy')
data_13 = np.load('./data_master/input_va_origin/input_1_005_20.npy')
data_14 = np.load('./data_master/input_va_origin/input_1_005_29.npy')
data_15 = np.load('./data_master/input_va_origin/input_1_005_5.npy')
data_16 = np.load('./data_master/input_va_origin/input_1_005_9.npy')
data_17 = np.load('./data_master/input_va_origin/input_1_006_11.npy')
data_18 = np.load('./data_master/input_va_origin/input_1_006_33.npy')
data_19 = np.load('./data_master/input_va_origin/input_1_006_5.npy')
data_20 = np.load('./data_master/input_va_origin/input_1_007_36.npy')
data_21 = np.load('./data_master/input_va_origin/input_1_007_8.npy')
data_22 = np.load('./data_master/input_va_origin/input_1_008_17.npy')
data_23 = np.load('./data_master/input_va_origin/input_1_008_22.npy')
data_24 = np.load('./data_master/input_va_origin/input_1_008_32.npy')
data_25 = np.load('./data_master/input_va_origin/input_1_009_17.npy')
data_26 = np.load('./data_master/input_va_origin/input_1_009_27.npy')
data_27 = np.load('./data_master/input_va_origin/input_1_010_14.npy')
data_28 = np.load('./data_master/input_va_origin/input_1_010_20.npy')
data_29 = np.load('./data_master/input_va_origin/input_1_010_29.npy')
data_30 = np.load('./data_master/input_va_origin/input_1_010_36.npy')
data_31 = np.load('./data_master/input_va_origin/input_1_011_13.npy')
data_32 = np.load('./data_master/input_va_origin/input_1_011_24.npy')
data_33 = np.load('./data_master/input_va_origin/input_1_011_30.npy')
data_34 = np.load('./data_master/input_va_origin/input_1_011_8.npy')
data_35 = np.load('./data_master/input_va_origin/input_1_012_13.npy')
data_36 = np.load('./data_master/input_va_origin/input_1_012_34.npy')
data_37 = np.load('./data_master/input_va_origin/input_1_012_9.npy')
data_38 = np.load('./data_master/input_va_origin/input_1_013_13.npy')
data_39 = np.load('./data_master/input_va_origin/input_1_013_18.npy')
data_40 = np.load('./data_master/input_va_origin/input_1_013_19.npy')
data_41 = np.load('./data_master/input_va_origin/input_1_013_21.npy')
data_42 = np.load('./data_master/input_va_origin/input_1_013_27.npy')
data_43 = np.load('./data_master/input_va_origin/input_1_013_37.npy')
data_44 = np.load('./data_master/input_va_origin/input_1_013_4.npy')
data_45 = np.load('./data_master/input_va_origin/input_1_014_15.npy')
data_46 = np.load('./data_master/input_va_origin/input_1_014_30.npy')
data_47 = np.load('./data_master/input_va_origin/input_1_014_5.npy')
data_48 = np.load('./data_master/input_va_origin/input_1_015_7.npy')
data_49 = np.load('./data_master/input_va_origin/input_1_016_7.npy')
data_50 = np.load('./data_master/input_va_origin/input_2_001_14.npy')
data_51 = np.load('./data_master/input_va_origin/input_2_001_6.npy')
data_52 = np.load('./data_master/input_va_origin/input_2_001_8.npy')
data_53 = np.load('./data_master/input_va_origin/input_2_002_18.npy')
data_54 = np.load('./data_master/input_va_origin/input_2_002_25.npy')
data_55 = np.load('./data_master/input_va_origin/input_2_002_35.npy')
data_56 = np.load('./data_master/input_va_origin/input_2_002_5.npy')
data_57 = np.load('./data_master/input_va_origin/input_2_003_20.npy')
data_58 = np.load('./data_master/input_va_origin/input_2_003_32.npy')
data_59 = np.load('./data_master/input_va_origin/input_2_003_5.npy')

data_dp_1 = np.concatenate((
data_0,data_1,data_2,data_3,data_4
),axis=0)
np.save("input_dp_1", data_dp_1)

data_dp_2 = np.concatenate((
data_5,data_6,data_7,data_8,data_9,
),axis=0)
np.save("input_dp_2", data_dp_2)

data_dp_3 = np.concatenate((
data_10,data_11,data_12,data_13,data_14
),axis=0)
np.save("input_dp_3", data_dp_3)

data_dp_4 = np.concatenate((
data_15,data_16,data_17,data_18,data_19,
),axis=0)
np.save("input_dp_4", data_dp_4)

data_dp_5 = np.concatenate((
data_20,data_21,data_22,data_23,data_24
),axis=0)
np.save("input_dp_5", data_dp_5)

data_dp_6 = np.concatenate((
data_25,data_26,data_27,data_28,data_29,
),axis=0)
np.save("input_dp_6", data_dp_6)

data_dp_7 = np.concatenate((
data_30,data_31,data_32,data_33,data_34
),axis=0)
np.save("input_dp_7", data_dp_7)

data_dp_8 = np.concatenate((
data_35,data_36,data_37,data_38,data_39,
),axis=0)
np.save("input_dp_8", data_dp_8)

data_dp_9 = np.concatenate((
data_40,data_41,data_42,data_43,data_44
),axis=0)
np.save("input_dp_9", data_dp_9)

data_dp_10 = np.concatenate((
data_45,data_46,data_47,data_48,data_49,
),axis=0)
np.save("input_dp_10", data_dp_10)

data_dp_pre = np.concatenate((
data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,
data_10,data_11,data_12,data_13,data_14,data_15,data_16,data_17,data_18,data_19,
data_20,data_21,data_22,data_23,data_24,data_25,data_26,data_27,data_28,data_29,
data_30,data_31,data_32,data_33,data_34,data_35,data_36,data_37,data_38,data_39,
),axis=0)
np.save("input_dp_pre", data_dp_pre)

data_dp_vld = np.concatenate((
data_40,data_41,data_42,data_43,data_44,data_45,data_46,data_47,data_48,data_49,
),axis=0)
np.save("input_dp_vld", data_dp_vld)
print("===========End==========")
'''

# ? input for pfc (previous)
data_1 = np.load('./data_master/input_pfc_12.9/input_02_0.6.npy')
data_2 = np.load('./data_master/input_pfc_12.9/input_03_0.6.npy')
data_3 = np.load('./data_master/input_pfc_12.9/input_04_0.6.npy')
data_4 = np.load('./data_master/input_pfc_12.9/input_05_0.6.npy')
data_5 = np.load('./data_master/input_pfc_12.9/input_06_0.6.npy')

# * prediction
data_p = np.concatenate((
data_1,
data_2,
data_3,
data_4,
),axis=0)
np.save("input_pfc_pre", data_p)
print("===========End==========")

# * validation
data_v = np.concatenate((
data_5,
),axis=0)
np.save("input_pfc_vld", data_v)
print("===========End==========")
