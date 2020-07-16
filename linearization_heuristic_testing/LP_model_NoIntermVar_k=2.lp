\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.4607391889289885 u_vars_time_2[0] - 333633.4095054139 u_vars_time_2[2]
   - 333681.152808971 u_vars_time_2[3] + -0 u_vars_time_2[4]
   + 186628.6941499851 u_vars_time_2[5] + -0 u_vars_time_2[6]
   + 1.6466469863246087e+07 u_vars_time_2[7]
   + 222443.3618433357 u_vars_time_2[8]
   - 1.417086761284136 u_vars_time_2[10]
   - 438545.7281610357 u_vars_time_2[12]
   - 438608.4845314313 u_vars_time_2[13] + -0 u_vars_time_2[14]
   + 186628.6941499851 u_vars_time_2[15] + -0 u_vars_time_2[16]
   + 1.9418057691834681e+07 u_vars_time_2[17]
   + 222443.3618433357 u_vars_time_2[18]
   + 5.8045722571871076e+06 u_vars_time_2[19]
   - 333786.1720555538 u_vars_time_3[2]
   - 333786.1720555538 u_vars_time_3[3] + -0 u_vars_time_3[4]
   + 186621.4673035756 u_vars_time_3[5] + -0 u_vars_time_3[6]
   + 1.6466102479173712e+07 u_vars_time_3[7]
   + 222434.748141042 u_vars_time_3[8]
   - 438746.5274871168 u_vars_time_3[12]
   - 438746.5274871168 u_vars_time_3[13] + -0 u_vars_time_3[14]
   + 186621.4673035756 u_vars_time_3[15] + -0 u_vars_time_3[16]
   + 1.941730576394305e+07 u_vars_time_3[17]
   + 222434.748141042 u_vars_time_3[18]
   + 5.8043474860051274e+06 u_vars_time_3[19]
   + 8.9289549039747566e+07 Constant
Subject To
 Const0_time_2: u_vars_time_2[2] <= 27.68181636881371
 Const1_time_2: u_vars_time_2[12] <= 46.98150772693063
 Const2_time_2: - u_vars_time_2[3] - u_vars_time_2[13]
   <= 2411.680551026998
 Const3_time_2: u_vars_time_2[3] <= 7.898924465137074
 Const4_time_2: u_vars_time_2[13] <= 13.4060343385329
 Const5_time_2: u_vars_time_2[0] + u_vars_time_2[10] <= 30000
 Const6_time_2: u_vars_time_2[1] + u_vars_time_2[11] <= 30000
 R7: - u_vars_time_2[8] + 0.25 u_vars_time_2[9] <= 0
 R8: - u_vars_time_2[18] + 0.25 u_vars_time_2[19] <= 0
 Const0_time_3: u_vars_time_3[2] <= 28.72727820814689
 Const1_time_3: u_vars_time_3[12] <= 49.96215676753138
 Const2_time_3: - 0.9475616151022548 u_vars_time_2[3]
   - 0.9475616151022548 u_vars_time_2[13] - u_vars_time_3[3]
   - u_vars_time_3[13] <= 2393.858082665514
 Const3_time_3: u_vars_time_3[3] <= 8.197243910293842
 Const4_time_3: u_vars_time_3[13] <= 14.25655373058093
 Const5_time_3: u_vars_time_3[0] + u_vars_time_3[10] <= 30000
 Const6_time_3: u_vars_time_3[1] + u_vars_time_3[11] <= 30000
 R16: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= 0
 R17: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= 0
Bounds
 u_vars_time_2[4] = 1
 u_vars_time_2[5] <= 1
 u_vars_time_2[6] <= 1
 u_vars_time_2[7] <= 1
 u_vars_time_2[8] <= 1
 u_vars_time_2[9] <= 1
 u_vars_time_2[14] = 1
 u_vars_time_2[15] <= 1
 u_vars_time_2[16] <= 1
 u_vars_time_2[17] <= 1
 u_vars_time_2[18] <= 1
 u_vars_time_2[19] <= 1
 u_vars_time_3[4] = 1
 u_vars_time_3[5] <= 1
 u_vars_time_3[6] <= 1
 u_vars_time_3[7] <= 1
 u_vars_time_3[8] <= 1
 u_vars_time_3[9] <= 1
 u_vars_time_3[14] = 1
 u_vars_time_3[15] <= 1
 u_vars_time_3[16] <= 1
 u_vars_time_3[17] <= 1
 u_vars_time_3[18] <= 1
 u_vars_time_3[19] <= 1
 Constant = 1
End
