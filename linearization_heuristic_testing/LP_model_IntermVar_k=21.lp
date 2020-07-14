\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  178598.8352554566 u_vars_time_21[5] + -0 u_vars_time_21[6]
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
 Initial_Conditions: x_vars_time_21[0] = 1.1410536729906958e+06
 Initial_Conditions: x_vars_time_21[1] = 43593.03088344837
 Initial_Conditions: x_vars_time_21[2] = 48122.71080862526
 Initial_Conditions: x_vars_time_21[3] = 351308.9772967272
 Initial_Conditions: x_vars_time_21[4] = 1.5840783919794967e+06
 Initial_Conditions: x_vars_time_21[5] = -0
 Initial_Conditions: x_vars_time_21[6] = -0
 Initial_Conditions: x_vars_time_21[7] = -0
 Initial_Conditions: x_vars_time_21[8] = -0
 Initial_Conditions: x_vars_time_21[9] = 10443.06109711896
 Initial_Conditions: x_vars_time_21[10] = 242.9997510430175
 Initial_Conditions: x_vars_time_21[11] = 89.46436207892188
 Initial_Conditions: x_vars_time_21[12] = 2.082805820677521
 Initial_Conditions: x_vars_time_21[13] = 680356.9768414185
 Initial_Conditions: x_vars_time_21[14] = 83851.58622264578
 Initial_Conditions: x_vars_time_21[15] = 96035.34274634987
 Initial_Conditions: x_vars_time_21[16] = 616214.172609865
 Initial_Conditions: x_vars_time_21[17] = 1.4764580784202782e+06
 Initial_Conditions: x_vars_time_21[18] = 7.114796047570207
 Initial_Conditions: x_vars_time_21[19] = 3.557398023785104
 Initial_Conditions: x_vars_time_21[20] = 3.557398023785104
 Initial_Conditions: x_vars_time_21[21] = 0.0285162166235279
 Initial_Conditions: x_vars_time_21[22] = 68981.9622895928
 Initial_Conditions: x_vars_time_21[23] = 477.28599569373
 Initial_Conditions: x_vars_time_21[24] = 173.5540683591742
 Initial_Conditions: x_vars_time_21[25] = 3.861109940117114
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
 R59: - u_vars_time_21[8] + 0.25 u_vars_time_21[9] <= -0
 R60: - u_vars_time_21[18] + 0.25 u_vars_time_21[19] <= -0
Bounds
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
