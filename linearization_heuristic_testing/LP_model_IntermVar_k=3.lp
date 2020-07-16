\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  186621.4673035756 u_vars_time_3[5] + -0 u_vars_time_3[6]
   + 1.6466102479173712e+07 u_vars_time_3[7]
   + 222434.748141042 u_vars_time_3[8]
   + 186621.4673035756 u_vars_time_3[15] + -0 u_vars_time_3[16]
   + 1.941730576394305e+07 u_vars_time_3[17]
   + 222434.748141042 u_vars_time_3[18]
   + 5.8043474860051274e+06 u_vars_time_3[19]
   + 10.32536320952022 x_vars_time_3[0]
   + 10.32536320952022 x_vars_time_3[1]
   + 10.32536320952022 x_vars_time_3[2]
   + 10.32536320952022 x_vars_time_3[3]
   + 10.32536320952022 x_vars_time_3[9]
   + 18.29768039680393 x_vars_time_3[13]
   + 18.29768039680393 x_vars_time_3[14]
   + 18.29768039680393 x_vars_time_3[15]
   + 18.29768039680393 x_vars_time_3[16]
   + 18.29768039680393 x_vars_time_3[22]
   - 333786.1720555538 x_vars_time_4[12]
   - 438746.5274871168 x_vars_time_4[25]
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
 Initial_Conditions: x_vars_time_3[16] = 148773.6203216587
 Initial_Conditions: x_vars_time_3[17] = 1.5427803468757053e+06
 Initial_Conditions: x_vars_time_3[18] = 481.9973882892103
 Initial_Conditions: x_vars_time_3[19] = 240.9986941446052
 Initial_Conditions: x_vars_time_3[20] = 240.9986941446052
 Initial_Conditions: x_vars_time_3[21] = 1.931853259676194
 Initial_Conditions: x_vars_time_3[22] = 2170.746538971613
 Initial_Conditions: x_vars_time_3[23] = 136.1052289011503
 Initial_Conditions: x_vars_time_3[24] = 55.67582972177269
 Initial_Conditions: x_vars_time_3[25] = 0.1988901079316664
 Dynamics_const_time_3[0]: 7350.405693642978 u_vars_time_3[4]
   + 6098.409974509981 u_vars_time_3[5]
   + 7905.093704478611 u_vars_time_3[7]
   + 452.2463282358231 u_vars_time_3[8]
   + 1507.105854391537 u_vars_time_3[14]
   + 960.973707691597 u_vars_time_3[15]
   + 531.4850086277355 u_vars_time_3[17]
   + 116.7381783988985 u_vars_time_3[18]
   - 0.9877396658138845 x_vars_time_3[0]
   + 0.1405890476857398 x_vars_time_3[2] - 0.00651037 x_vars_time_3[4]
   - 0.00651037 x_vars_time_3[9] + 0.0488941118106405 x_vars_time_3[15]
   - 0.00403417656700325 x_vars_time_3[17]
   - 0.00403417656700325 x_vars_time_3[22] + x_vars_time_4[0]
   = 24922.45844997728
 Dynamics_const_time_3[1]: - 7350.405693642978 u_vars_time_3[4]
   - 6098.409974509981 u_vars_time_3[5]
   - 7905.093704478611 u_vars_time_3[7]
   - 452.2463282358231 u_vars_time_3[8]
   - 1507.105854391537 u_vars_time_3[14]
   - 960.973707691597 u_vars_time_3[15]
   - 531.4850086277355 u_vars_time_3[17]
   - 116.7381783988985 u_vars_time_3[18]
   - 0.0122603341861155 x_vars_time_3[0] - 0.75 x_vars_time_3[1]
   - 0.1405890476857398 x_vars_time_3[2] + 0.00651037 x_vars_time_3[4]
   + 0.00651037 x_vars_time_3[9] - 0.0488941118106405 x_vars_time_3[15]
   + 0.00403417656700325 x_vars_time_3[17]
   + 0.00403417656700325 x_vars_time_3[22] + x_vars_time_4[1]
   = -24922.45844997716
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
   + 412.342879249484 u_vars_time_3[5] + 228.0541642665985 u_vars_time_3[7]
   + 50.09102281032532 u_vars_time_3[8]
   + 9862.266142379263 u_vars_time_3[14]
   + 17918.46258985733 u_vars_time_3[15]
   + 24730.9524382759 u_vars_time_3[17]
   + 3149.286613221538 u_vars_time_3[18]
   + 43.19727297066193 u_vars_time_3[19]
   + 0.0362136041314144 x_vars_time_3[2]
   - 0.00167697248449678 x_vars_time_3[4]
   - 0.00167697248449678 x_vars_time_3[9]
   - 0.965927087142799 x_vars_time_3[13]
   + 0.2773421371150129 x_vars_time_3[15]
   - 0.0228830652436251 x_vars_time_3[17]
   - 0.0228830652436251 x_vars_time_3[22] + x_vars_time_4[13]
   = 57041.33508344356
 Dynamics_const_time_3[14]: - 646.6819604124588 u_vars_time_3[4]
   - 412.342879249484 u_vars_time_3[5] - 228.0541642665985 u_vars_time_3[7]
   - 50.09102281032532 u_vars_time_3[8]
   - 9862.266142379263 u_vars_time_3[14]
   - 17918.46258985733 u_vars_time_3[15]
   - 24730.9524382759 u_vars_time_3[17]
   - 3149.286613221538 u_vars_time_3[18]
   - 43.19727297066193 u_vars_time_3[19]
   - 0.0362136041314144 x_vars_time_3[2]
   + 0.00167697248449678 x_vars_time_3[4]
   + 0.00167697248449678 x_vars_time_3[9]
   - 0.034072912857201 x_vars_time_3[13] - 0.75 x_vars_time_3[14]
   - 0.2773421371150129 x_vars_time_3[15]
   + 0.0228830652436251 x_vars_time_3[17]
   + 0.0228830652436251 x_vars_time_3[22] + x_vars_time_4[14]
   = -57041.33508344357
 Dynamics_const_time_3[15]: 0.0826245256653223 u_vars_time_3[10]
   - 0.25 x_vars_time_3[14] - 0.75 x_vars_time_3[15] + x_vars_time_4[15]
   = 7.27596e-12
 Dynamics_const_time_3[16]: 0.096432146431565 u_vars_time_3[11]
   - 0.2495 x_vars_time_3[15] - x_vars_time_3[16] + x_vars_time_4[16]
   = -7.27596e-12
 Dynamics_const_time_3[17]: 0.0826245256653223 u_vars_time_3[10]
   + 0.096432146431565 u_vars_time_3[11] + 0.0005 x_vars_time_3[15]
   - x_vars_time_3[17] + x_vars_time_4[17] = -1.41611e-11
 Dynamics_const_time_3[18]: - 0.0412296383069958 u_vars_time_3[10]
   - 0.75 x_vars_time_3[18] + x_vars_time_4[18] = -2.84217e-14
 Dynamics_const_time_3[19]: - 0.0206148191534979 u_vars_time_3[10]
   - 0.75 x_vars_time_3[19] + x_vars_time_4[19] = -1.42109e-14
 Dynamics_const_time_3[20]: - 0.0206148191534979 u_vars_time_3[10]
   - 0.75 x_vars_time_3[20] + x_vars_time_4[20] = -1.42109e-14
 Dynamics_const_time_3[21]: - 0.000165249 u_vars_time_3[10]
   - 0.75 x_vars_time_3[21] + x_vars_time_4[21] = -0
 Dynamics_const_time_3[22]: - 0.096432146431565 u_vars_time_3[11]
   - 0.25 x_vars_time_3[18] - 0.25 x_vars_time_3[19]
   - 0.25 x_vars_time_3[20] - x_vars_time_3[22]
   - 0.0758199847444699 x_vars_time_3[23]
   - 0.0521237545883587 x_vars_time_3[24] + x_vars_time_4[22]
   = -1.42109e-13
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
