\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  -0 u_vars_time_3[5] + -0 u_vars_time_3[6] + -0 u_vars_time_3[7]
   + -0 u_vars_time_3[15] + -0 u_vars_time_3[16] + -0 u_vars_time_3[17]
   - 1.1159708999999999e+06 x_vars_time_4[12]
   - 1.1159708999999999e+06 x_vars_time_4[25]
Subject To
 Initial_Conditions: x_vars_time_3[0] = 1.3551810291436231e+06
 Initial_Conditions: x_vars_time_3[1] = 77468.89735272036
 Initial_Conditions: x_vars_time_3[2] = 73849.04423688145
 Initial_Conditions: x_vars_time_3[3] = 88224.83171777232
 Initial_Conditions: x_vars_time_3[4] = 1.5947238024509971e+06
 Initial_Conditions: x_vars_time_3[5] = -0
 Initial_Conditions: x_vars_time_3[6] = -0
 Initial_Conditions: x_vars_time_3[7] = -0
 Initial_Conditions: x_vars_time_3[8] = -0
 Initial_Conditions: x_vars_time_3[9] = 19.15141460344243
 Initial_Conditions: x_vars_time_3[10] = 80.2872607247908
 Initial_Conditions: x_vars_time_3[11] = 32.64361922992402
 Initial_Conditions: x_vars_time_3[12] = 0.1152505503601564
 Initial_Conditions: x_vars_time_3[13] = 1.1160641557611616e+06
 Initial_Conditions: x_vars_time_3[14] = 150471.0764264984
 Initial_Conditions: x_vars_time_3[15] = 127471.4943663865
 Initial_Conditions: x_vars_time_3[16] = 146980.5359350268
 Initial_Conditions: x_vars_time_3[17] = 1.5409872624890734e+06
 Initial_Conditions: x_vars_time_3[18] = 481.9973882892103
 Initial_Conditions: x_vars_time_3[19] = 240.9986941446052
 Initial_Conditions: x_vars_time_3[20] = 240.9986941446052
 Initial_Conditions: x_vars_time_3[21] = 1.931853259676194
 Initial_Conditions: x_vars_time_3[22] = 3963.83092560358
 Initial_Conditions: x_vars_time_3[23] = 136.1052289011503
 Initial_Conditions: x_vars_time_3[24] = 55.67582972177269
 Initial_Conditions: x_vars_time_3[25] = 0.1988901079316664
 Dynamics_const_time_3[0]: 7350.405693642978 u_vars_time_3[4]
   + 1507.105854391537 u_vars_time_3[14]
   - 0.9956426428855625 x_vars_time_3[0]
   + 0.0391446635468153 x_vars_time_3[2]
   - 0.00181270340959928 x_vars_time_3[4]
   - 0.00181270340959928 x_vars_time_3[9]
   + 0.023646162804991 x_vars_time_3[15]
   - 0.0019510078484886 x_vars_time_3[17]
   - 0.0019510078484886 x_vars_time_3[22] + x_vars_time_4[0]
   = 8857.511548034601
 Dynamics_const_time_3[1]: - 7350.405693642978 u_vars_time_3[4]
   - 1507.105854391537 u_vars_time_3[14]
   - 0.00435735711443748 x_vars_time_3[0] - 0.75 x_vars_time_3[1]
   - 0.0391446635468153 x_vars_time_3[2]
   + 0.00181270340959928 x_vars_time_3[4]
   + 0.00181270340959928 x_vars_time_3[9]
   - 0.023646162804991 x_vars_time_3[15]
   + 0.0019510078484886 x_vars_time_3[17]
   + 0.0019510078484886 x_vars_time_3[22] + x_vars_time_4[1]
   = -8857.511548034519
 Dynamics_const_time_3[2]: 0.046308360183362 u_vars_time_3[0]
   - 0.25 x_vars_time_3[1] - 0.75 x_vars_time_3[2] + x_vars_time_4[2]
   = -3.63798e-12
 Dynamics_const_time_3[3]: 0.0553229540953587 u_vars_time_3[1]
   - 0.2495 x_vars_time_3[2] - x_vars_time_3[3] + x_vars_time_4[3]
   = -7.27596e-12
 Dynamics_const_time_3[4]: 0.046308360183362 u_vars_time_3[0]
   + 0.0553229540953587 u_vars_time_3[1] + 0.0005 x_vars_time_3[2]
   - x_vars_time_3[4] + x_vars_time_4[4] = 3.06315e-11
 Dynamics_const_time_3[5]: - 0.0231078717314977 u_vars_time_3[0]
   - 0.75 x_vars_time_3[5] + x_vars_time_4[5] = -0
 Dynamics_const_time_3[6]: - 0.0115539358657488 u_vars_time_3[0]
   - 0.75 x_vars_time_3[6] + x_vars_time_4[6] = -0
 Dynamics_const_time_3[7]: - 0.0115539358657488 u_vars_time_3[0]
   - 0.75 x_vars_time_3[7] + x_vars_time_4[7] = -0
 Dynamics_const_time_3[8]: - 9.26167e-05 u_vars_time_3[0]
   - 0.75 x_vars_time_3[8] + x_vars_time_4[8] = -0
 Dynamics_const_time_3[9]: - 0.0553229540953587 u_vars_time_3[1]
   - 0.25 x_vars_time_3[5] - 0.25 x_vars_time_3[6] - 0.25 x_vars_time_3[7]
   - x_vars_time_3[9] - 0.0758199847444699 x_vars_time_3[10]
   - 0.0521237545883587 x_vars_time_3[11] + x_vars_time_4[9] = -0
 Dynamics_const_time_3[10]: u_vars_time_3[2] - 0.000389 x_vars_time_3[2]
   - 0.1945 x_vars_time_3[8] - 0.92372234935164 x_vars_time_3[10]
   + x_vars_time_4[10] = -2.87273e-08
 Dynamics_const_time_3[11]: u_vars_time_3[3] - 0.000111 x_vars_time_3[2]
   - 0.0555 x_vars_time_3[8] - 0.9475616151022549 x_vars_time_3[11]
   + x_vars_time_4[11] = -8.19724e-09
 Dynamics_const_time_3[12]: - u_vars_time_3[2] - u_vars_time_3[3]
   - 0.000457666 x_vars_time_3[10]
   - 3.1463030938647091e-04 x_vars_time_3[11] - x_vars_time_3[12]
   + x_vars_time_4[12] = -1.38778e-17
 Dynamics_const_time_3[13]: 646.6819604124588 u_vars_time_3[4]
   + 9862.266142379262 u_vars_time_3[14]
   + 0.0175136175991157 x_vars_time_3[2]
   - 8.1101717219684156e-04 x_vars_time_3[4]
   - 8.1101717219684156e-04 x_vars_time_3[9]
   - 0.9937226140937364 x_vars_time_3[13]
   + 0.0448147368902471 x_vars_time_3[15]
   - 0.00369759373315186 x_vars_time_3[17]
   - 0.00369759373315186 x_vars_time_3[22] + x_vars_time_4[13]
   = 10508.94810279175
 Dynamics_const_time_3[14]: - 646.6819604124588 u_vars_time_3[4]
   - 9862.266142379262 u_vars_time_3[14]
   - 0.0175136175991157 x_vars_time_3[2]
   + 8.1101717219684156e-04 x_vars_time_3[4]
   + 8.1101717219684156e-04 x_vars_time_3[9]
   - 0.00627738590626364 x_vars_time_3[13] - 0.75 x_vars_time_3[14]
   - 0.0448147368902471 x_vars_time_3[15]
   + 0.00369759373315186 x_vars_time_3[17]
   + 0.00369759373315186 x_vars_time_3[22] + x_vars_time_4[14]
   = -10508.94810279171
 Dynamics_const_time_3[15]: 0.0827206671134249 u_vars_time_3[10]
   - 0.25 x_vars_time_3[14] - 0.75 x_vars_time_3[15] + x_vars_time_4[15]
   = 7.27596e-12
 Dynamics_const_time_3[16]: 0.0953807597978565 u_vars_time_3[11]
   - 0.2495 x_vars_time_3[15] - x_vars_time_3[16] + x_vars_time_4[16]
   = -7.27596e-12
 Dynamics_const_time_3[17]: 0.0827206671134249 u_vars_time_3[10]
   + 0.0953807597978565 u_vars_time_3[11] + 0.0005 x_vars_time_3[15]
   - x_vars_time_3[17] + x_vars_time_4[17] = -1.41611e-11
 Dynamics_const_time_3[18]: - 0.041277612889599 u_vars_time_3[10]
   - 0.75 x_vars_time_3[18] + x_vars_time_4[18] = -2.84217e-14
 Dynamics_const_time_3[19]: - 0.0206388064447995 u_vars_time_3[10]
   - 0.75 x_vars_time_3[19] + x_vars_time_4[19] = -1.42109e-14
 Dynamics_const_time_3[20]: - 0.0206388064447995 u_vars_time_3[10]
   - 0.75 x_vars_time_3[20] + x_vars_time_4[20] = -1.42109e-14
 Dynamics_const_time_3[21]: - 1.6544133422684976e-04 u_vars_time_3[10]
   - 0.75 x_vars_time_3[21] + x_vars_time_4[21] = -0
 Dynamics_const_time_3[22]: - 0.0953807597978565 u_vars_time_3[11]
   - 0.25 x_vars_time_3[18] - 0.25 x_vars_time_3[19]
   - 0.25 x_vars_time_3[20] - x_vars_time_3[22]
   - 0.0758199847444699 x_vars_time_3[23]
   - 0.0521237545883587 x_vars_time_3[24] + x_vars_time_4[22]
   = 3.12639e-13
 Dynamics_const_time_3[23]: u_vars_time_3[12] - 0.000389 x_vars_time_3[15]
   - 0.1945 x_vars_time_3[21] - 0.92372234935164 x_vars_time_3[23]
   + x_vars_time_4[23] = -4.99621e-08
 Dynamics_const_time_3[24]: u_vars_time_3[13] - 0.000111 x_vars_time_3[15]
   - 0.0555 x_vars_time_3[21] - 0.9475616151022549 x_vars_time_3[24]
   + x_vars_time_4[24] = -1.42566e-08
 Dynamics_const_time_3[25]: - u_vars_time_3[12] - u_vars_time_3[13]
   - 0.000457666 x_vars_time_3[23]
   - 3.1463030938647091e-04 x_vars_time_3[24] - x_vars_time_3[25]
   + x_vars_time_4[25] = -2.77556e-17
 All_const_time_3[0]: u_vars_time_3[2] - 0.000389 x_vars_time_3[2]
   - 0.1945 x_vars_time_3[8] <= -0
 All_const_time_3[1]: u_vars_time_3[12] - 0.000389 x_vars_time_3[15]
   - 0.1945 x_vars_time_3[21] <= -0
 All_const_time_3[2]: - u_vars_time_3[3] - u_vars_time_3[13]
   + 0.000111 x_vars_time_3[2] + 0.0555 x_vars_time_3[8]
   + 0.9475616151022548 x_vars_time_3[11] + 0.000111 x_vars_time_3[15]
   + 0.0555 x_vars_time_3[21] + 0.9475616151022548 x_vars_time_3[24]
   <= 2500
 All_const_time_3[3]: u_vars_time_3[3] - 0.000111 x_vars_time_3[2]
   - 0.0555 x_vars_time_3[8] <= -0
 All_const_time_3[4]: u_vars_time_3[13] - 0.000111 x_vars_time_3[15]
   - 0.0555 x_vars_time_3[21] <= -0
 All_const_time_3[5]: u_vars_time_3[0] + u_vars_time_3[10] <= 30000
 All_const_time_3[6]: u_vars_time_3[1] + u_vars_time_3[11] <= 30000
 R59: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= -0
 R60: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= -0
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
End
