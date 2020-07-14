\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  178604.8172442199 u_vars_time_20[5] + -0 u_vars_time_20[6]
   + 1.6356438891678147e+07 u_vars_time_20[7]
   + 212879.6762479089 u_vars_time_20[8]
   + 178604.8172442199 u_vars_time_20[15] + -0 u_vars_time_20[16]
   + 1.8583201586893432e+07 u_vars_time_20[17]
   + 212879.6762479089 u_vars_time_20[18]
   + 5.5550116336483788e+06 u_vars_time_20[19]
   + 10.32536320952022 x_vars_time_20[0]
   + 10.32536320952022 x_vars_time_20[1]
   + 10.32536320952022 x_vars_time_20[2]
   + 10.32536320952022 x_vars_time_20[3]
   + 10.32536320952022 x_vars_time_20[9]
   + 18.29768039680393 x_vars_time_20[13]
   + 18.29768039680393 x_vars_time_20[14]
   + 18.29768039680393 x_vars_time_20[15]
   + 18.29768039680393 x_vars_time_20[16]
   + 18.29768039680393 x_vars_time_20[22]
   + 178598.8352554566 u_vars_time_21[5] + -0 u_vars_time_21[6]
   + 1.6356184749541037e+07 u_vars_time_21[7]
   + 212872.5462955878 u_vars_time_21[8]
   + 178598.8352554566 u_vars_time_21[15] + -0 u_vars_time_21[16]
   + 1.8582579182051301e+07 u_vars_time_21[17]
   + 212872.5462955878 u_vars_time_21[18]
   + 5.5548255803398201e+06 u_vars_time_21[19]
   + 10.32536320952022 x_vars_time_21[0]
   + 10.32536320952022 x_vars_time_21[1]
   + 10.32536320952022 x_vars_time_21[2]
   + 10.32536320952022 x_vars_time_21[3]
   + 10.32536320952022 x_vars_time_21[9]
   + 18.29768039680393 x_vars_time_21[13]
   + 18.29768039680393 x_vars_time_21[14]
   + 18.29768039680393 x_vars_time_21[15]
   + 18.29768039680393 x_vars_time_21[16]
   + 18.29768039680393 x_vars_time_21[22] - 1e+07 x_vars_time_22[12]
   - 1e+07 x_vars_time_22[25]
Subject To
 Initial_Conditions: x_vars_time_20[0] = 1.1510388027145185e+06
 Initial_Conditions: x_vars_time_20[1] = 44810.53487950087
 Initial_Conditions: x_vars_time_20[2] = 49226.76945166673
 Initial_Conditions: x_vars_time_20[3] = 339026.8983185363
 Initial_Conditions: x_vars_time_20[4] = 1.5841030053642225e+06
 Initial_Conditions: x_vars_time_20[5] = -0
 Initial_Conditions: x_vars_time_20[6] = -0
 Initial_Conditions: x_vars_time_20[7] = -0
 Initial_Conditions: x_vars_time_20[8] = -0
 Initial_Conditions: x_vars_time_20[9] = 10420.06653017437
 Initial_Conditions: x_vars_time_20[10] = 242.3353055196606
 Initial_Conditions: x_vars_time_20[11] = 88.64879004853556
 Initial_Conditions: x_vars_time_20[12] = 1.94400561779266
 Initial_Conditions: x_vars_time_20[13] = 698638.7025400599
 Initial_Conditions: x_vars_time_20[14] = 87426.48069867252
 Initial_Conditions: x_vars_time_20[15] = 98904.96342890899
 Initial_Conditions: x_vars_time_20[16] = 591537.3842343522
 Initial_Conditions: x_vars_time_20[17] = 1.4765075309019927e+06
 Initial_Conditions: x_vars_time_20[18] = 9.486394730093609
 Initial_Conditions: x_vars_time_20[19] = 4.743197365046805
 Initial_Conditions: x_vars_time_20[20] = 4.743197365046805
 Initial_Conditions: x_vars_time_20[21] = 0.0380216221647039
 Initial_Conditions: x_vars_time_20[22] = 68932.25871488295
 Initial_Conditions: x_vars_time_20[23] = 475.039464034676
 Initial_Conditions: x_vars_time_20[24] = 171.5703809002138
 Initial_Conditions: x_vars_time_20[25] = 3.589719332401998
 Dynamics_const_time_20[0]: 4441.167621347547 u_vars_time_20[4]
   + 3631.124694570786 u_vars_time_20[5]
   + 4574.654947341445 u_vars_time_20[7]
   + 277.6953206216118 u_vars_time_20[8]
   + 992.8966921607196 u_vars_time_20[14]
   + 633.0992695968438 u_vars_time_20[15]
   + 350.147738768185 u_vars_time_20[17]
   + 76.90830132693644 u_vars_time_20[18]
   - 0.9913251145832143 x_vars_time_20[0]
   + 0.1194274128813117 x_vars_time_20[2]
   - 0.00368701201239608 x_vars_time_20[4]
   - 0.00368701201239608 x_vars_time_20[9]
   + 0.0415156516048536 x_vars_time_20[15]
   - 0.00265691619388375 x_vars_time_20[17]
   - 0.00265691619388375 x_vars_time_20[22] + x_vars_time_21[0]
   = 14977.6945857341
 Dynamics_const_time_20[1]: - 4441.167621347547 u_vars_time_20[4]
   - 3631.124694570786 u_vars_time_20[5]
   - 4574.654947341445 u_vars_time_20[7]
   - 277.6953206216118 u_vars_time_20[8]
   - 992.8966921607196 u_vars_time_20[14]
   - 633.0992695968438 u_vars_time_20[15]
   - 350.147738768185 u_vars_time_20[17]
   - 76.90830132693644 u_vars_time_20[18]
   - 0.00867488541678576 x_vars_time_20[0] - 0.75 x_vars_time_20[1]
   - 0.1194274128813117 x_vars_time_20[2]
   + 0.00368701201239608 x_vars_time_20[4]
   + 0.00368701201239608 x_vars_time_20[9]
   - 0.0415156516048536 x_vars_time_20[15]
   + 0.00265691619388375 x_vars_time_20[17]
   + 0.00265691619388375 x_vars_time_20[22] + x_vars_time_21[1]
   = -14977.69458573408
 Dynamics_const_time_20[2]: 0.0310754851704535 u_vars_time_20[0]
   - 0.25 x_vars_time_20[1] - 0.75 x_vars_time_20[2] + x_vars_time_21[2]
   = 1.81899e-12
 Dynamics_const_time_20[3]: 0.2140182154635747 u_vars_time_20[1]
   - 0.2495 x_vars_time_20[2] - x_vars_time_20[3] + x_vars_time_21[3]
   = -9.09495e-12
 Dynamics_const_time_20[4]: 0.0310754851704535 u_vars_time_20[0]
   + 0.2140182154635747 u_vars_time_20[1] + 0.0005 x_vars_time_20[2]
   - x_vars_time_20[4] + x_vars_time_21[4] = 7.45572e-11
 Dynamics_const_time_20[5]: - 0.0155066671000563 u_vars_time_20[0]
   - 0.75 x_vars_time_20[5] + x_vars_time_21[5] = -0
 Dynamics_const_time_20[6]: - 0.00775333355002815 u_vars_time_20[0]
   - 0.75 x_vars_time_20[6] + x_vars_time_21[6] = -0
 Dynamics_const_time_20[7]: - 0.00775333355002815 u_vars_time_20[0]
   - 0.75 x_vars_time_20[7] + x_vars_time_21[7] = -0
 Dynamics_const_time_20[8]: - 6.2151e-05 u_vars_time_20[0]
   - 0.75 x_vars_time_20[8] + x_vars_time_21[8] = -0
 Dynamics_const_time_20[9]: - 0.2140182154635747 u_vars_time_20[1]
   - 0.25 x_vars_time_20[5] - 0.25 x_vars_time_20[6]
   - 0.25 x_vars_time_20[7] - x_vars_time_20[9]
   - 0.0758199847444699 x_vars_time_20[10]
   - 0.0521237545883587 x_vars_time_20[11] + x_vars_time_21[9]
   = -4.15668e-13
 Dynamics_const_time_20[10]: u_vars_time_20[2] - 0.000389 x_vars_time_20[2]
   - 0.1945 x_vars_time_20[8] - 0.92372234935164 x_vars_time_20[10]
   + x_vars_time_21[10] = -1.91492e-08
 Dynamics_const_time_20[11]: u_vars_time_20[3] - 0.000111 x_vars_time_20[2]
   - 0.0555 x_vars_time_20[8] - 0.9475616151022549 x_vars_time_20[11]
   + x_vars_time_21[11] = -5.46418e-09
 Dynamics_const_time_20[12]: - u_vars_time_20[2] - u_vars_time_20[3]
   - 0.000457666 x_vars_time_20[10]
   - 3.1463030938647091e-04 x_vars_time_20[11] - x_vars_time_20[12]
   + x_vars_time_21[12] = 5.55112e-17
 Dynamics_const_time_20[13]: 269.8799168289907 u_vars_time_20[4]
   + 172.0831394243627 u_vars_time_20[5]
   + 95.17389173113673 u_vars_time_20[7]
   + 20.90449695134132 u_vars_time_20[8]
   + 4700.363271144092 u_vars_time_20[14]
   + 8643.995685849388 u_vars_time_20[15]
   + 11976.92175012315 u_vars_time_20[17]
   + 1522.292096312699 u_vars_time_20[18]
   + 20.97429959693958 u_vars_time_20[19]
   + 0.0226722757211904 x_vars_time_20[2]
   - 6.9994778347464359e-04 x_vars_time_20[4]
   - 6.9994778347464359e-04 x_vars_time_20[9]
   - 0.9738323605145635 x_vars_time_20[13]
   + 0.1735569400529436 x_vars_time_20[15]
   - 0.0111072866921753 x_vars_time_20[17]
   - 0.0111072866921753 x_vars_time_20[22] + x_vars_time_21[13]
   = 27422.58854796211
 Dynamics_const_time_20[14]: - 269.8799168289907 u_vars_time_20[4]
   - 172.0831394243627 u_vars_time_20[5]
   - 95.17389173113673 u_vars_time_20[7]
   - 20.90449695134132 u_vars_time_20[8]
   - 4700.363271144092 u_vars_time_20[14]
   - 8643.995685849388 u_vars_time_20[15]
   - 11976.92175012315 u_vars_time_20[17]
   - 1522.292096312699 u_vars_time_20[18]
   - 20.97429959693958 u_vars_time_20[19]
   - 0.0226722757211904 x_vars_time_20[2]
   + 6.9994778347464359e-04 x_vars_time_20[4]
   + 6.9994778347464359e-04 x_vars_time_20[9]
   - 0.0261676394854365 x_vars_time_20[13] - 0.75 x_vars_time_20[14]
   - 0.1735569400529436 x_vars_time_20[15]
   + 0.0111072866921753 x_vars_time_20[17]
   + 0.0111072866921753 x_vars_time_20[22] + x_vars_time_21[14]
   = -27422.5885479621
 Dynamics_const_time_20[15]: 0.0669857493842164 u_vars_time_20[10]
   - 0.25 x_vars_time_20[14] - 0.75 x_vars_time_20[15] + x_vars_time_21[15]
   = -3.63798e-12
 Dynamics_const_time_20[16]: 0.4006328256740989 u_vars_time_20[11]
   - 0.2495 x_vars_time_20[15] - x_vars_time_20[16] + x_vars_time_21[16]
   = 2.18279e-11
 Dynamics_const_time_20[17]: 0.0669857493842164 u_vars_time_20[10]
   + 0.4006328256740989 u_vars_time_20[11] + 0.0005 x_vars_time_20[15]
   - x_vars_time_20[17] + x_vars_time_21[17] = -8.84626e-11
 Dynamics_const_time_20[18]: - 0.033425888942724 u_vars_time_20[10]
   - 0.75 x_vars_time_20[18] + x_vars_time_21[18] = 4.44089e-16
 Dynamics_const_time_20[19]: - 0.016712944471362 u_vars_time_20[10]
   - 0.75 x_vars_time_20[19] + x_vars_time_21[19] = 2.22045e-16
 Dynamics_const_time_20[20]: - 0.016712944471362 u_vars_time_20[10]
   - 0.75 x_vars_time_20[20] + x_vars_time_21[20] = 2.22045e-16
 Dynamics_const_time_20[21]: - 1.3397149876843273e-04 u_vars_time_20[10]
   - 0.75 x_vars_time_20[21] + x_vars_time_21[21] = -1.73472e-18
 Dynamics_const_time_20[22]: - 0.4006328256740989 u_vars_time_20[11]
   - 0.25 x_vars_time_20[18] - 0.25 x_vars_time_20[19]
   - 0.25 x_vars_time_20[20] - x_vars_time_20[22]
   - 0.0758199847444699 x_vars_time_20[23]
   - 0.0521237545883587 x_vars_time_20[24] + x_vars_time_21[22]
   = 5.13722e-12
 Dynamics_const_time_20[23]: u_vars_time_20[12]
   - 0.000389 x_vars_time_20[15] - 0.1945 x_vars_time_20[21]
   - 0.92372234935164 x_vars_time_20[23] + x_vars_time_21[23]
   = -3.84814e-08
 Dynamics_const_time_20[24]: u_vars_time_20[13]
   - 0.000111 x_vars_time_20[15] - 0.0555 x_vars_time_20[21]
   - 0.9475616151022549 x_vars_time_20[24] + x_vars_time_21[24]
   = -1.09806e-08
 Dynamics_const_time_20[25]: - u_vars_time_20[12] - u_vars_time_20[13]
   - 0.000457666 x_vars_time_20[23]
   - 3.1463030938647091e-04 x_vars_time_20[24] - x_vars_time_20[25]
   + x_vars_time_21[25] = 1.11022e-16
 All_const_time_20[0]: u_vars_time_20[2] - 0.000389 x_vars_time_20[2]
   - 0.1945 x_vars_time_20[8] <= -0
 All_const_time_20[1]: u_vars_time_20[12] - 0.000389 x_vars_time_20[15]
   - 0.1945 x_vars_time_20[21] <= -0
 All_const_time_20[2]: - u_vars_time_20[3] - u_vars_time_20[13]
   + 0.000111 x_vars_time_20[2] + 0.0555 x_vars_time_20[8]
   + 0.9475616151022548 x_vars_time_20[11] + 0.000111 x_vars_time_20[15]
   + 0.0555 x_vars_time_20[21] + 0.9475616151022548 x_vars_time_20[24]
   <= 2500
 All_const_time_20[3]: u_vars_time_20[3] - 0.000111 x_vars_time_20[2]
   - 0.0555 x_vars_time_20[8] <= -0
 All_const_time_20[4]: u_vars_time_20[13] - 0.000111 x_vars_time_20[15]
   - 0.0555 x_vars_time_20[21] <= -0
 All_const_time_20[5]: u_vars_time_20[0] + u_vars_time_20[10] <= 30000
 All_const_time_20[6]: u_vars_time_20[1] + u_vars_time_20[11] <= 30000
 R59: - u_vars_time_20[8] + 0.25 u_vars_time_20[9] <= -0
 R60: - u_vars_time_20[18] + 0.25 u_vars_time_20[19] <= -0
 Dynamics_const_time_21[0]: 4290.935817858956 u_vars_time_21[4]
   + 3510.628067961324 u_vars_time_21[5]
   + 4428.691595507594 u_vars_time_21[7]
   + 268.1080836069373 u_vars_time_21[8]
   + 955.7253513725698 u_vars_time_21[14]
   + 609.3977617877073 u_vars_time_21[15]
   + 337.039163599475 u_vars_time_21[17]
   + 74.02906454366138 u_vars_time_21[18]
   - 0.9915431644459514 x_vars_time_21[0]
   + 0.1183915139543115 x_vars_time_21[2]
   - 0.003573059852676 x_vars_time_21[4]
   - 0.003573059852676 x_vars_time_21[9]
   + 0.0411555013975004 x_vars_time_21[15]
   - 0.0025574480914778 x_vars_time_21[17]
   - 0.0025574480914778 x_vars_time_21[22] + x_vars_time_22[0]
   = 14474.55490623815
 Dynamics_const_time_21[1]: - 4290.935817858956 u_vars_time_21[4]
   - 3510.628067961324 u_vars_time_21[5]
   - 4428.691595507594 u_vars_time_21[7]
   - 268.1080836069373 u_vars_time_21[8]
   - 955.7253513725698 u_vars_time_21[14]
   - 609.3977617877073 u_vars_time_21[15]
   - 337.039163599475 u_vars_time_21[17]
   - 74.02906454366138 u_vars_time_21[18]
   - 0.00845683555404862 x_vars_time_21[0] - 0.75 x_vars_time_21[1]
   - 0.1183915139543115 x_vars_time_21[2]
   + 0.003573059852676 x_vars_time_21[4]
   + 0.003573059852676 x_vars_time_21[9]
   - 0.0411555013975004 x_vars_time_21[15]
   + 0.0025574480914778 x_vars_time_21[17]
   + 0.0025574480914778 x_vars_time_21[22] + x_vars_time_22[1]
   = -14474.55490623823
 Dynamics_const_time_21[2]: 0.0303789957948294 u_vars_time_21[0]
   - 0.25 x_vars_time_21[1] - 0.75 x_vars_time_21[2] + x_vars_time_22[2]
   = -0
 Dynamics_const_time_21[3]: 0.2217749948963854 u_vars_time_21[1]
   - 0.2495 x_vars_time_21[2] - x_vars_time_21[3] + x_vars_time_22[3]
   = -2.36469e-11
 Dynamics_const_time_21[4]: 0.0303789957948294 u_vars_time_21[0]
   + 0.2217749948963854 u_vars_time_21[1] + 0.0005 x_vars_time_21[2]
   - x_vars_time_21[4] + x_vars_time_22[4] = -1.03348e-11
 Dynamics_const_time_21[5]: - 0.0151591189016199 u_vars_time_21[0]
   - 0.75 x_vars_time_21[5] + x_vars_time_22[5] = -0
 Dynamics_const_time_21[6]: - 0.00757955945080994 u_vars_time_21[0]
   - 0.75 x_vars_time_21[6] + x_vars_time_22[6] = -0
 Dynamics_const_time_21[7]: - 0.00757955945080994 u_vars_time_21[0]
   - 0.75 x_vars_time_21[7] + x_vars_time_22[7] = -0
 Dynamics_const_time_21[8]: - 6.0758e-05 u_vars_time_21[0]
   - 0.75 x_vars_time_21[8] + x_vars_time_22[8] = -0
 Dynamics_const_time_21[9]: - 0.2217749948963854 u_vars_time_21[1]
   - 0.25 x_vars_time_21[5] - 0.25 x_vars_time_21[6]
   - 0.25 x_vars_time_21[7] - x_vars_time_21[9]
   - 0.0758199847444699 x_vars_time_21[10]
   - 0.0521237545883587 x_vars_time_21[11] + x_vars_time_22[9]
   = 7.67386e-13
 Dynamics_const_time_21[10]: u_vars_time_21[2] - 0.000389 x_vars_time_21[2]
   - 0.1945 x_vars_time_21[8] - 0.92372234935164 x_vars_time_21[10]
   + x_vars_time_22[10] = -1.87197e-08
 Dynamics_const_time_21[11]: u_vars_time_21[3] - 0.000111 x_vars_time_21[2]
   - 0.0555 x_vars_time_21[8] - 0.9475616151022549 x_vars_time_21[11]
   + x_vars_time_22[11] = -5.34162e-09
 Dynamics_const_time_21[12]: - u_vars_time_21[2] - u_vars_time_21[3]
   - 0.000457666 x_vars_time_21[10]
   - 3.1463030938647091e-04 x_vars_time_21[11] - x_vars_time_21[12]
   + x_vars_time_22[12] = 1.94289e-16
 Dynamics_const_time_21[13]: 256.9235762483428 u_vars_time_21[4]
   + 163.8218067962593 u_vars_time_21[5]
   + 90.60480274466104 u_vars_time_21[7]
   + 19.90091808059373 u_vars_time_21[8]
   + 4448.020202742887 u_vars_time_21[14]
   + 8175.775661089994 u_vars_time_21[15]
   + 11326.32890154707 u_vars_time_21[17]
   + 1439.713365909284 u_vars_time_21[18]
   + 19.83282604600426 u_vars_time_21[19]
   + 0.0220790181992257 x_vars_time_21[2]
   - 6.6634550804545449e-04 x_vars_time_21[4]
   - 6.6634550804545449e-04 x_vars_time_21[9]
   - 0.9745810675432607 x_vars_time_21[13]
   + 0.169015337154931 x_vars_time_21[15]
   - 0.0105027988181334 x_vars_time_21[17]
   - 0.0105027988181334 x_vars_time_21[22] + x_vars_time_22[13]
   = 25940.92206120507
 Dynamics_const_time_21[14]: - 256.9235762483428 u_vars_time_21[4]
   - 163.8218067962593 u_vars_time_21[5]
   - 90.60480274466104 u_vars_time_21[7]
   - 19.90091808059373 u_vars_time_21[8]
   - 4448.020202742887 u_vars_time_21[14]
   - 8175.775661089994 u_vars_time_21[15]
   - 11326.32890154707 u_vars_time_21[17]
   - 1439.713365909284 u_vars_time_21[18]
   - 19.83282604600426 u_vars_time_21[19]
   - 0.0220790181992257 x_vars_time_21[2]
   + 6.6634550804545449e-04 x_vars_time_21[4]
   + 6.6634550804545449e-04 x_vars_time_21[9]
   - 0.0254189324567394 x_vars_time_21[13] - 0.75 x_vars_time_21[14]
   - 0.169015337154931 x_vars_time_21[15]
   + 0.0105027988181334 x_vars_time_21[17]
   + 0.0105027988181334 x_vars_time_21[22] + x_vars_time_22[14]
   = -25940.92206120509
 Dynamics_const_time_21[15]: 0.0650444087441358 u_vars_time_21[10]
   - 0.25 x_vars_time_21[14] - 0.75 x_vars_time_21[15] + x_vars_time_22[15]
   = -3.63798e-12
 Dynamics_const_time_21[16]: 0.4173597487232264 u_vars_time_21[11]
   - 0.2495 x_vars_time_21[15] - x_vars_time_21[16] + x_vars_time_22[16]
   = -3.63798e-12
 Dynamics_const_time_21[17]: 0.0650444087441358 u_vars_time_21[10]
   + 0.4173597487232264 u_vars_time_21[11] + 0.0005 x_vars_time_21[15]
   - x_vars_time_21[17] + x_vars_time_22[17] = -3.21876e-11
 Dynamics_const_time_21[18]: - 0.0324571599633238 u_vars_time_21[10]
   - 0.75 x_vars_time_21[18] + x_vars_time_22[18] = -4.44089e-16
 Dynamics_const_time_21[19]: - 0.0162285799816619 u_vars_time_21[10]
   - 0.75 x_vars_time_21[19] + x_vars_time_22[19] = -2.22045e-16
 Dynamics_const_time_21[20]: - 0.0162285799816619 u_vars_time_21[10]
   - 0.75 x_vars_time_21[20] + x_vars_time_22[20] = -2.22045e-16
 Dynamics_const_time_21[21]: - 1.3008881748827159e-04 u_vars_time_21[10]
   - 0.75 x_vars_time_21[21] + x_vars_time_22[21] = -0
 Dynamics_const_time_21[22]: - 0.4173597487232264 u_vars_time_21[11]
   - 0.25 x_vars_time_21[18] - 0.25 x_vars_time_21[19]
   - 0.25 x_vars_time_21[20] - x_vars_time_21[22]
   - 0.0758199847444699 x_vars_time_21[23]
   - 0.0521237545883587 x_vars_time_21[24] + x_vars_time_22[22]
   = -1.93978e-12
 Dynamics_const_time_21[23]: u_vars_time_21[12]
   - 0.000389 x_vars_time_21[15] - 0.1945 x_vars_time_21[21]
   - 0.92372234935164 x_vars_time_21[23] + x_vars_time_22[23]
   = -3.73633e-08
 Dynamics_const_time_21[24]: u_vars_time_21[13]
   - 0.000111 x_vars_time_21[15] - 0.0555 x_vars_time_21[21]
   - 0.9475616151022549 x_vars_time_21[24] + x_vars_time_22[24]
   = -1.06615e-08
 Dynamics_const_time_21[25]: - u_vars_time_21[12] - u_vars_time_21[13]
   - 0.000457666 x_vars_time_21[23]
   - 3.1463030938647091e-04 x_vars_time_21[24] - x_vars_time_21[25]
   + x_vars_time_22[25] = 5.55112e-17
 All_const_time_21[0]: u_vars_time_21[2] - 0.000389 x_vars_time_21[2]
   - 0.1945 x_vars_time_21[8] <= -0
 All_const_time_21[1]: u_vars_time_21[12] - 0.000389 x_vars_time_21[15]
   - 0.1945 x_vars_time_21[21] <= -0
 All_const_time_21[2]: - u_vars_time_21[3] - u_vars_time_21[13]
   + 0.000111 x_vars_time_21[2] + 0.0555 x_vars_time_21[8]
   + 0.9475616151022548 x_vars_time_21[11] + 0.000111 x_vars_time_21[15]
   + 0.0555 x_vars_time_21[21] + 0.9475616151022548 x_vars_time_21[24]
   <= 2500
 All_const_time_21[3]: u_vars_time_21[3] - 0.000111 x_vars_time_21[2]
   - 0.0555 x_vars_time_21[8] <= -0
 All_const_time_21[4]: u_vars_time_21[13] - 0.000111 x_vars_time_21[15]
   - 0.0555 x_vars_time_21[21] <= -0
 All_const_time_21[5]: u_vars_time_21[0] + u_vars_time_21[10] <= 30000
 All_const_time_21[6]: u_vars_time_21[1] + u_vars_time_21[11] <= 30000
 R94: - u_vars_time_21[8] + 0.25 u_vars_time_21[9] <= -0
 R95: - u_vars_time_21[18] + 0.25 u_vars_time_21[19] <= -0
Bounds
 u_vars_time_20[4] = 1
 u_vars_time_20[5] <= 1
 u_vars_time_20[6] <= 1
 u_vars_time_20[7] <= 1
 u_vars_time_20[8] <= 1
 u_vars_time_20[9] <= 1
 u_vars_time_20[14] = 1
 u_vars_time_20[15] <= 1
 u_vars_time_20[16] <= 1
 u_vars_time_20[17] <= 1
 u_vars_time_20[18] <= 1
 u_vars_time_20[19] <= 1
 u_vars_time_21[4] = 1
 u_vars_time_21[5] <= 1
 u_vars_time_21[6] <= 1
 u_vars_time_21[7] <= 1
 u_vars_time_21[8] <= 1
 u_vars_time_21[9] <= 1
 u_vars_time_21[14] = 1
 u_vars_time_21[15] <= 1
 u_vars_time_21[16] <= 1
 u_vars_time_21[17] <= 1
 u_vars_time_21[18] <= 1
 u_vars_time_21[19] <= 1
End
