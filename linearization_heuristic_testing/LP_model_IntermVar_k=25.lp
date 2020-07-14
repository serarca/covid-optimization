\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  176496.5839462549 u_vars_time_25[5] + -0 u_vars_time_25[6]
   + 1.6199121969032703e+07 u_vars_time_25[7]
   + 210366.8659617666 u_vars_time_25[8]
   + 176496.5839462549 u_vars_time_25[15] + -0 u_vars_time_25[16]
   + 1.8363847344534367e+07 u_vars_time_25[17]
   + 210366.8659617666 u_vars_time_25[18]
   + 5.4894408350700475e+06 u_vars_time_25[19]
   + 10.32536320952022 x_vars_time_25[0]
   + 10.32536320952022 x_vars_time_25[1]
   + 10.32536320952022 x_vars_time_25[2]
   + 10.32536320952022 x_vars_time_25[3]
   + 10.32536320952022 x_vars_time_25[9]
   + 18.29768039680393 x_vars_time_25[13]
   + 18.29768039680393 x_vars_time_25[14]
   + 18.29768039680393 x_vars_time_25[15]
   + 18.29768039680393 x_vars_time_25[16]
   + 18.29768039680393 x_vars_time_25[22]
Subject To
 Initial_Conditions: x_vars_time_25[0] = 1.0482546794878298e+06
 Initial_Conditions: x_vars_time_25[1] = 43458.50719981828
 Initial_Conditions: x_vars_time_25[2] = 50724.6929648428
 Initial_Conditions: x_vars_time_25[3] = 426429.1558525411
 Initial_Conditions: x_vars_time_25[4] = 1.5688670355050322e+06
 Initial_Conditions: x_vars_time_25[5] = -0
 Initial_Conditions: x_vars_time_25[6] = -0
 Initial_Conditions: x_vars_time_25[7] = -0
 Initial_Conditions: x_vars_time_25[8] = -0
 Initial_Conditions: x_vars_time_25[9] = 25604.00900052648
 Initial_Conditions: x_vars_time_25[10] = 278.2842990088845
 Initial_Conditions: x_vars_time_25[11] = 103.8021405282866
 Initial_Conditions: x_vars_time_25[12] = 2.869050281465488
 Initial_Conditions: x_vars_time_25[13] = 512221.013005694
 Initial_Conditions: x_vars_time_25[14] = 69067.75506460061
 Initial_Conditions: x_vars_time_25[15] = 90451.57352665639
 Initial_Conditions: x_vars_time_25[16] = 787338.6429232992
 Initial_Conditions: x_vars_time_25[17] = 1.4590789845202502e+06
 Initial_Conditions: x_vars_time_25[18] = 0.8597901781562326
 Initial_Conditions: x_vars_time_25[19] = 0.4298950890781163
 Initial_Conditions: x_vars_time_25[20] = 0.4298950890781163
 Initial_Conditions: x_vars_time_25[21] = 0.00344605281826145
 Initial_Conditions: x_vars_time_25[22] = 86269.61070163223
 Initial_Conditions: x_vars_time_25[23] = 549.3720168191637
 Initial_Conditions: x_vars_time_25[24] = 203.8004177826575
 Initial_Conditions: x_vars_time_25[25] = 5.509308906105804
 Dynamics_const_time_25[0]: 3958.231622993707 u_vars_time_25[4]
   + 3273.989912473283 u_vars_time_25[5]
   + 4219.162017201735 u_vars_time_25[7]
   + 244.3695444656219 u_vars_time_25[8]
   + 826.9982239321465 u_vars_time_25[14]
   + 527.3176712775055 u_vars_time_25[15]
   + 291.6431894288891 u_vars_time_25[17]
   + 64.05805267071973 u_vars_time_25[18]
   - 0.9914742282279508 x_vars_time_25[0]
   + 0.1087664716965589 x_vars_time_25[2]
   - 0.00346017314060919 x_vars_time_25[4]
   - 0.00346017314060919 x_vars_time_25[9]
   + 0.0378106664292648 x_vars_time_25[15]
   - 0.00221311507655492 x_vars_time_25[17]
   - 0.00221311507655492 x_vars_time_25[22] + x_vars_time_26[0]
   = 13405.77023444358
 Dynamics_const_time_25[1]: - 3958.231622993707 u_vars_time_25[4]
   - 3273.989912473283 u_vars_time_25[5]
   - 4219.162017201735 u_vars_time_25[7]
   - 244.3695444656219 u_vars_time_25[8]
   - 826.9982239321465 u_vars_time_25[14]
   - 527.3176712775055 u_vars_time_25[15]
   - 291.6431894288891 u_vars_time_25[17]
   - 64.05805267071973 u_vars_time_25[18]
   - 0.00852577177204912 x_vars_time_25[0] - 0.75 x_vars_time_25[1]
   - 0.1087664716965589 x_vars_time_25[2]
   + 0.00346017314060919 x_vars_time_25[4]
   + 0.00346017314060919 x_vars_time_25[9]
   - 0.0378106664292648 x_vars_time_25[15]
   + 0.00221311507655492 x_vars_time_25[17]
   + 0.00221311507655492 x_vars_time_25[22] + x_vars_time_26[1]
   = -13405.77023444361
 Dynamics_const_time_25[2]: 0.0323320535245449 u_vars_time_25[0]
   - 0.25 x_vars_time_25[1] - 0.75 x_vars_time_25[2] + x_vars_time_26[2]
   = -1.81899e-12
 Dynamics_const_time_25[3]: 0.2718070723662505 u_vars_time_25[1]
   - 0.2495 x_vars_time_25[2] - x_vars_time_25[3] + x_vars_time_26[3]
   = 9.09495e-12
 Dynamics_const_time_25[4]: 0.0323320535245449 u_vars_time_25[0]
   + 0.2718070723662505 u_vars_time_25[1] + 0.0005 x_vars_time_25[2]
   - x_vars_time_25[4] + x_vars_time_26[4] = -4.17764e-11
 Dynamics_const_time_25[5]: - 0.0161336947087479 u_vars_time_25[0]
   - 0.75 x_vars_time_25[5] + x_vars_time_26[5] = -0
 Dynamics_const_time_25[6]: - 0.00806684735437396 u_vars_time_25[0]
   - 0.75 x_vars_time_25[6] + x_vars_time_26[6] = -0
 Dynamics_const_time_25[7]: - 0.00806684735437396 u_vars_time_25[0]
   - 0.75 x_vars_time_25[7] + x_vars_time_26[7] = -0
 Dynamics_const_time_25[8]: - 6.46641e-05 u_vars_time_25[0]
   - 0.75 x_vars_time_25[8] + x_vars_time_26[8] = -0
 Dynamics_const_time_25[9]: - 0.2718070723662505 u_vars_time_25[1]
   - 0.25 x_vars_time_25[5] - 0.25 x_vars_time_25[6]
   - 0.25 x_vars_time_25[7] - x_vars_time_25[9]
   - 0.0758199847444699 x_vars_time_25[10]
   - 0.0521237545883587 x_vars_time_25[11] + x_vars_time_26[9]
   = 5.04485e-13
 Dynamics_const_time_25[10]: u_vars_time_25[2] - 0.000389 x_vars_time_25[2]
   - 0.1945 x_vars_time_25[8] - 0.92372234935164 x_vars_time_25[10]
   + x_vars_time_26[10] = -1.97319e-08
 Dynamics_const_time_25[11]: u_vars_time_25[3] - 0.000111 x_vars_time_25[2]
   - 0.0555 x_vars_time_25[8] - 0.9475616151022549 x_vars_time_25[11]
   + x_vars_time_26[11] = -5.63044e-09
 Dynamics_const_time_25[12]: - u_vars_time_25[2] - u_vars_time_25[3]
   - 0.000457666 x_vars_time_25[10]
   - 3.1463030938647091e-04 x_vars_time_25[11] - x_vars_time_25[12]
   + x_vars_time_26[12] = 8.32667e-17
 Dynamics_const_time_25[13]: 203.8954712203945 u_vars_time_25[4]
   + 130.0095731993578 u_vars_time_25[5]
   + 71.90429629002456 u_vars_time_25[7]
   + 15.79343993654728 u_vars_time_25[8]
   + 3197.665917034953 u_vars_time_25[14]
   + 5825.434525179132 u_vars_time_25[15]
   + 8047.245735740851 u_vars_time_25[17]
   + 1024.318540023539 u_vars_time_25[18]
   + 14.06422756562882 u_vars_time_25[19]
   + 0.0166231772339573 x_vars_time_25[2] - 0.000528831 x_vars_time_25[4]
   - 0.000528831 x_vars_time_25[9] - 0.9758823746576957 x_vars_time_25[13]
   + 0.1272542695947922 x_vars_time_25[15]
   - 0.007448383464044 x_vars_time_25[17]
   - 0.007448383464044 x_vars_time_25[22] + x_vars_time_26[13]
   = 18530.33172619041
 Dynamics_const_time_25[14]: - 203.8954712203945 u_vars_time_25[4]
   - 130.0095731993578 u_vars_time_25[5]
   - 71.90429629002456 u_vars_time_25[7]
   - 15.79343993654728 u_vars_time_25[8]
   - 3197.665917034953 u_vars_time_25[14]
   - 5825.434525179132 u_vars_time_25[15]
   - 8047.245735740851 u_vars_time_25[17]
   - 1024.318540023539 u_vars_time_25[18]
   - 14.06422756562882 u_vars_time_25[19]
   - 0.0166231772339573 x_vars_time_25[2] + 0.000528831 x_vars_time_25[4]
   + 0.000528831 x_vars_time_25[9] - 0.0241176253423044 x_vars_time_25[13]
   - 0.75 x_vars_time_25[14] - 0.1272542695947922 x_vars_time_25[15]
   + 0.007448383464044 x_vars_time_25[17]
   + 0.007448383464044 x_vars_time_25[22] + x_vars_time_26[14]
   = -18530.33172619043
 Dynamics_const_time_25[15]: 0.0619922392730488 u_vars_time_25[10]
   - 0.25 x_vars_time_25[14] - 0.75 x_vars_time_25[15] + x_vars_time_26[15]
   = 3.63798e-12
 Dynamics_const_time_25[16]: 0.5396134488100921 u_vars_time_25[11]
   - 0.2495 x_vars_time_25[15] - x_vars_time_25[16] + x_vars_time_26[16]
   = 5.09317e-11
 Dynamics_const_time_25[17]: 0.0619922392730488 u_vars_time_25[10]
   + 0.5396134488100921 u_vars_time_25[11] + 0.0005 x_vars_time_25[15]
   - x_vars_time_25[17] + x_vars_time_26[17] = 8.48175e-11
 Dynamics_const_time_25[18]: - 0.0309341273972513 u_vars_time_25[10]
   - 0.75 x_vars_time_25[18] + x_vars_time_26[18] = -2.77556e-17
 Dynamics_const_time_25[19]: - 0.0154670636986257 u_vars_time_25[10]
   - 0.75 x_vars_time_25[19] + x_vars_time_26[19] = -1.38778e-17
 Dynamics_const_time_25[20]: - 0.0154670636986257 u_vars_time_25[10]
   - 0.75 x_vars_time_25[20] + x_vars_time_26[20] = -1.38778e-17
 Dynamics_const_time_25[21]: - 1.2398447854609755e-04 u_vars_time_25[10]
   - 0.75 x_vars_time_25[21] + x_vars_time_26[21] = -0
 Dynamics_const_time_25[22]: - 0.5396134488100921 u_vars_time_25[11]
   - 0.25 x_vars_time_25[18] - 0.25 x_vars_time_25[19]
   - 0.25 x_vars_time_25[20] - x_vars_time_25[22]
   - 0.0758199847444699 x_vars_time_25[23]
   - 0.0521237545883587 x_vars_time_25[24] + x_vars_time_26[22]
   = -1.3074e-12
 Dynamics_const_time_25[23]: u_vars_time_25[12]
   - 0.000389 x_vars_time_25[15] - 0.1945 x_vars_time_25[21]
   - 0.92372234935164 x_vars_time_25[23] + x_vars_time_26[23]
   = -3.51864e-08
 Dynamics_const_time_25[24]: u_vars_time_25[13]
   - 0.000111 x_vars_time_25[15] - 0.0555 x_vars_time_25[21]
   - 0.9475616151022549 x_vars_time_25[24] + x_vars_time_26[24]
   = -1.00403e-08
 Dynamics_const_time_25[25]: - u_vars_time_25[12] - u_vars_time_25[13]
   - 0.000457666 x_vars_time_25[23]
   - 3.1463030938647091e-04 x_vars_time_25[24] - x_vars_time_25[25]
   + x_vars_time_26[25] = 4.44089e-16
 All_const_time_25[0]: u_vars_time_25[2] - 0.000389 x_vars_time_25[2]
   - 0.1945 x_vars_time_25[8] <= -0
 All_const_time_25[1]: u_vars_time_25[12] - 0.000389 x_vars_time_25[15]
   - 0.1945 x_vars_time_25[21] <= -0
 All_const_time_25[2]: - u_vars_time_25[3] - u_vars_time_25[13]
   + 0.000111 x_vars_time_25[2] + 0.0555 x_vars_time_25[8]
   + 0.9475616151022548 x_vars_time_25[11] + 0.000111 x_vars_time_25[15]
   + 0.0555 x_vars_time_25[21] + 0.9475616151022548 x_vars_time_25[24]
   <= 2500
 All_const_time_25[3]: u_vars_time_25[3] - 0.000111 x_vars_time_25[2]
   - 0.0555 x_vars_time_25[8] <= -0
 All_const_time_25[4]: u_vars_time_25[13] - 0.000111 x_vars_time_25[15]
   - 0.0555 x_vars_time_25[21] <= -0
 All_const_time_25[5]: u_vars_time_25[0] + u_vars_time_25[10] <= 30000
 All_const_time_25[6]: u_vars_time_25[1] + u_vars_time_25[11] <= 30000
 R59: - u_vars_time_25[8] + 0.25 u_vars_time_25[9] <= -0
 R60: - u_vars_time_25[18] + 0.25 u_vars_time_25[19] <= -0
Bounds
 u_vars_time_25[4] = 1
 u_vars_time_25[5] <= 1
 u_vars_time_25[6] <= 1
 u_vars_time_25[7] <= 1
 u_vars_time_25[8] <= 1
 u_vars_time_25[9] <= 1
 u_vars_time_25[14] = 1
 u_vars_time_25[15] <= 1
 u_vars_time_25[16] <= 1
 u_vars_time_25[17] <= 1
 u_vars_time_25[18] <= 1
 u_vars_time_25[19] <= 1
End
