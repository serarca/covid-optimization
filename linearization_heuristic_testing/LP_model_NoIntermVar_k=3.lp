\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 333786.1720555538 u_vars_time_3[2] - 333786.1720555538 u_vars_time_3[3]
   + -0 u_vars_time_3[4] + 186621.4673035756 u_vars_time_3[5]
   + -0 u_vars_time_3[6] + 1.6466102479173712e+07 u_vars_time_3[7]
   + 222434.748141042 u_vars_time_3[8]
   - 438746.5274871168 u_vars_time_3[12]
   - 438746.5274871168 u_vars_time_3[13] + -0 u_vars_time_3[14]
   + 186621.4673035756 u_vars_time_3[15] + -0 u_vars_time_3[16]
   + 1.941730576394305e+07 u_vars_time_3[17]
   + 222434.748141042 u_vars_time_3[18]
   + 5.8043474860051274e+06 u_vars_time_3[19]
   + 4.4558881586917609e+07 Constant
Subject To
 Const0_time_3: u_vars_time_3[2] <= 28.72727820814689
 Const1_time_3: u_vars_time_3[12] <= 49.96215676753138
 Const2_time_3: - u_vars_time_3[3] - u_vars_time_3[13]
   <= 2393.858082665514
 Const3_time_3: u_vars_time_3[3] <= 8.197243910293842
 Const4_time_3: u_vars_time_3[13] <= 14.25655373058093
 Const5_time_3: u_vars_time_3[0] + u_vars_time_3[10] <= 30000
 Const6_time_3: u_vars_time_3[1] + u_vars_time_3[11] <= 30000
 R7: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= 0
 R8: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= 0
Bounds
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
