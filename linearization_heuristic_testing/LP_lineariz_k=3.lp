\ Model Linearization Heuristic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 1.1159708999999999e+06 u[2] - 1.1159708999999999e+06 u[3] + -0 u[4]
   + -0 u[5] + -0 u[6] + -0 u[7] - 1.1159708999999999e+06 u[12]
   - 1.1159708999999999e+06 u[13] + -0 u[14] + -0 u[15] + -0 u[16]
   + -0 u[17]
Subject To
 transport_lock_lb[0]: u[8] - 0.25 u[9] >= 0
 transport_lock_lb[1]: u[18] - 0.25 u[19] >= 0
 R2: u[2] <= 28.72727820814689
 R3: u[12] <= 49.96215676753138
 R4: - u[3] - u[13] <= 2393.858082665514
 R5: u[3] <= 8.197243910293842
 R6: u[13] <= 14.25655373058093
 R7: u[0] + u[10] <= 30000
 R8: u[1] + u[11] <= 30000
Bounds
 u[4] = 1
 u[5] <= 1
 u[6] <= 1
 u[7] <= 1
 u[8] <= 1
 u[9] <= 1
 u[14] = 1
 u[15] <= 1
 u[16] <= 1
 u[17] <= 1
 u[18] <= 1
 u[19] <= 1
End
