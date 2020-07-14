\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  184549.9082621472 u_vars_time_7[5] + -0 u_vars_time_7[6]
   + 1.6440610872290853e+07 u_vars_time_7[7]
   + 219965.650023354 u_vars_time_7[8]
   + 184549.9082621472 u_vars_time_7[15] + -0 u_vars_time_7[16]
   + 1.9201767348686431e+07 u_vars_time_7[17]
   + 219965.650023354 u_vars_time_7[18]
   + 5.7399173393132295e+06 u_vars_time_7[19]
   + 10.32536320952022 x_vars_time_7[0]
   + 10.32536320952022 x_vars_time_7[1]
   + 10.32536320952022 x_vars_time_7[2]
   + 10.32536320952022 x_vars_time_7[3]
   + 10.32536320952022 x_vars_time_7[9]
   + 18.29768039680393 x_vars_time_7[13]
   + 18.29768039680393 x_vars_time_7[14]
   + 18.29768039680393 x_vars_time_7[15]
   + 18.29768039680393 x_vars_time_7[16]
   + 18.29768039680393 x_vars_time_7[22] - 1e+10 x_vars_time_8[12]
   - 1e+10 x_vars_time_8[25]
Subject To
 Initial_Conditions: x_vars_time_7[0] = 1.3093026592783511e+06
 Initial_Conditions: x_vars_time_7[1] = 61278.43257370069
 Initial_Conditions: x_vars_time_7[2] = 66129.73438301329
 Initial_Conditions: x_vars_time_7[3] = 155544.1423003025
 Initial_Conditions: x_vars_time_7[4] = 1.5922549685353674e+06
 Initial_Conditions: x_vars_time_7[5] = -0
 Initial_Conditions: x_vars_time_7[6] = -0
 Initial_Conditions: x_vars_time_7[7] = -0
 Initial_Conditions: x_vars_time_7[8] = -0
 Initial_Conditions: x_vars_time_7[9] = 2390.706812591739
 Initial_Conditions: x_vars_time_7[10] = 155.0440123903378
 Initial_Conditions: x_vars_time_7[11] = 54.91006652694409
 Initial_Conditions: x_vars_time_7[12] = 0.3705690897252158
 Initial_Conditions: x_vars_time_7[13] = 1.037742485720882e+06
 Initial_Conditions: x_vars_time_7[14] = 120574.6778082864
 Initial_Conditions: x_vars_time_7[15] = 114280.8095671285
 Initial_Conditions: x_vars_time_7[16] = 253057.0096790946
 Initial_Conditions: x_vars_time_7[17] = 1.5256549827753911e+06
 Initial_Conditions: x_vars_time_7[18] = 1241.116358299435
 Initial_Conditions: x_vars_time_7[19] = 620.5581791497174
 Initial_Conditions: x_vars_time_7[20] = 620.5581791497174
 Initial_Conditions: x_vars_time_7[21] = 4.97441426172118
 Initial_Conditions: x_vars_time_7[22] = 17607.08020992514
 Initial_Conditions: x_vars_time_7[23] = 265.0798949601671
 Initial_Conditions: x_vars_time_7[24] = 94.01633310364501
 Initial_Conditions: x_vars_time_7[25] = 0.6336487653284114
 Dynamics_const_time_7[0]: 6365.391081495757 u_vars_time_7[4]
   + 5280.068341620294 u_vars_time_7[5]
   + 6841.588413903884 u_vars_time_7[7]
   + 391.7332602905765 u_vars_time_7[8]
   + 1306.838077995972 u_vars_time_7[14]
   + 833.2772575363451 u_vars_time_7[15]
   + 460.860028600419 u_vars_time_7[17]
   + 101.2257342395897 u_vars_time_7[18]
   - 0.9890114624286195 x_vars_time_7[0]
   + 0.1358378247070555 x_vars_time_7[2]
   - 0.00563317569909929 x_vars_time_7[4]
   - 0.00563317569909929 x_vars_time_7[9]
   + 0.0472905487563081 x_vars_time_7[15]
   - 0.00350193419923138 x_vars_time_7[17]
   - 0.00350193419923138 x_vars_time_7[22] + x_vars_time_8[0]
   = 21580.98219568277
 Dynamics_const_time_7[1]: - 6365.391081495757 u_vars_time_7[4]
   - 5280.068341620294 u_vars_time_7[5]
   - 6841.588413903884 u_vars_time_7[7]
   - 391.7332602905765 u_vars_time_7[8]
   - 1306.838077995972 u_vars_time_7[14]
   - 833.2772575363451 u_vars_time_7[15]
   - 460.860028600419 u_vars_time_7[17]
   - 101.2257342395897 u_vars_time_7[18]
   - 0.0109885375713805 x_vars_time_7[0] - 0.75 x_vars_time_7[1]
   - 0.1358378247070555 x_vars_time_7[2]
   + 0.00563317569909929 x_vars_time_7[4]
   + 0.00563317569909929 x_vars_time_7[9]
   - 0.0472905487563081 x_vars_time_7[15]
   + 0.00350193419923138 x_vars_time_7[17]
   + 0.00350193419923138 x_vars_time_7[22] + x_vars_time_8[1]
   = -21580.98219568283
 Dynamics_const_time_7[2]: 0.041532126254781 u_vars_time_7[0]
   - 0.25 x_vars_time_7[1] - 0.75 x_vars_time_7[2] + x_vars_time_8[2] = -0
 Dynamics_const_time_7[3]: 0.0976879616481143 u_vars_time_7[1]
   - 0.2495 x_vars_time_7[2] - x_vars_time_7[3] + x_vars_time_8[3]
   = 7.27596e-12
 Dynamics_const_time_7[4]: 0.041532126254781 u_vars_time_7[0]
   + 0.0976879616481143 u_vars_time_7[1] + 0.0005 x_vars_time_7[2]
   - x_vars_time_7[4] + x_vars_time_8[4] = 2.43077e-11
 Dynamics_const_time_7[5]: - 0.0207245310011357 u_vars_time_7[0]
   - 0.75 x_vars_time_7[5] + x_vars_time_8[5] = -0
 Dynamics_const_time_7[6]: - 0.0103622655005679 u_vars_time_7[0]
   - 0.75 x_vars_time_7[6] + x_vars_time_8[6] = -0
 Dynamics_const_time_7[7]: - 0.0103622655005679 u_vars_time_7[0]
   - 0.75 x_vars_time_7[7] + x_vars_time_8[7] = -0
 Dynamics_const_time_7[8]: - 8.30643e-05 u_vars_time_7[0]
   - 0.75 x_vars_time_7[8] + x_vars_time_8[8] = -0
 Dynamics_const_time_7[9]: - 0.0976879616481143 u_vars_time_7[1]
   - 0.25 x_vars_time_7[5] - 0.25 x_vars_time_7[6] - 0.25 x_vars_time_7[7]
   - x_vars_time_7[9] - 0.0758199847444699 x_vars_time_7[10]
   - 0.0521237545883587 x_vars_time_7[11] + x_vars_time_8[9] = 4.79616e-14
 Dynamics_const_time_7[10]: u_vars_time_7[2] - 0.000389 x_vars_time_7[2]
   - 0.1945 x_vars_time_7[8] - 0.92372234935164 x_vars_time_7[10]
   + x_vars_time_8[10] = -2.57245e-08
 Dynamics_const_time_7[11]: u_vars_time_7[3] - 0.000111 x_vars_time_7[2]
   - 0.0555 x_vars_time_7[8] - 0.9475616151022549 x_vars_time_7[11]
   + x_vars_time_8[11] = -7.3404e-09
 Dynamics_const_time_7[12]: - u_vars_time_7[2] - u_vars_time_7[3]
   - 0.000457666 x_vars_time_7[10]
   - 3.1463030938647091e-04 x_vars_time_7[11] - x_vars_time_7[12]
   + x_vars_time_8[12] = -0
 Dynamics_const_time_7[13]: 538.4799970264019 u_vars_time_7[4]
   + 343.3502184760223 u_vars_time_7[5]
   + 189.8964455693369 u_vars_time_7[7]
   + 41.70985966076711 u_vars_time_7[8]
   + 8227.863391184084 u_vars_time_7[14]
   + 14951.76539541199 u_vars_time_7[15]
   + 20637.58796464854 u_vars_time_7[17]
   + 2627.95245489412 u_vars_time_7[18]
   + 36.04890312036741 u_vars_time_7[19]
   + 0.0336743079681426 x_vars_time_7[2]
   - 0.00139646886821998 x_vars_time_7[4]
   - 0.00139646886821998 x_vars_time_7[9]
   - 0.969424235598019 x_vars_time_7[13]
   + 0.2581614255587917 x_vars_time_7[15]
   - 0.0191172305854468 x_vars_time_7[17]
   - 0.0191172305854468 x_vars_time_7[22] + x_vars_time_8[13]
   = 47594.65462999167
 Dynamics_const_time_7[14]: - 538.4799970264019 u_vars_time_7[4]
   - 343.3502184760223 u_vars_time_7[5]
   - 189.8964455693369 u_vars_time_7[7]
   - 41.70985966076711 u_vars_time_7[8]
   - 8227.863391184084 u_vars_time_7[14]
   - 14951.76539541199 u_vars_time_7[15]
   - 20637.58796464854 u_vars_time_7[17]
   - 2627.95245489412 u_vars_time_7[18]
   - 36.04890312036741 u_vars_time_7[19]
   - 0.0336743079681426 x_vars_time_7[2]
   + 0.00139646886821998 x_vars_time_7[4]
   + 0.00139646886821998 x_vars_time_7[9]
   - 0.0305757644019809 x_vars_time_7[13] - 0.75 x_vars_time_7[14]
   - 0.2581614255587917 x_vars_time_7[15]
   + 0.0191172305854468 x_vars_time_7[17]
   + 0.0191172305854468 x_vars_time_7[22] + x_vars_time_8[14]
   = -47594.65462999162
 Dynamics_const_time_7[15]: 0.0749060638593628 u_vars_time_7[10]
   - 0.25 x_vars_time_7[14] - 0.75 x_vars_time_7[15] + x_vars_time_8[15]
   = -0
 Dynamics_const_time_7[16]: 0.1658677830414493 u_vars_time_7[11]
   - 0.2495 x_vars_time_7[15] - x_vars_time_7[16] + x_vars_time_8[16]
   = 1.09139e-11
 Dynamics_const_time_7[17]: 0.0749060638593628 u_vars_time_7[10]
   + 0.1658677830414493 u_vars_time_7[11] + 0.0005 x_vars_time_7[15]
   - x_vars_time_7[17] + x_vars_time_8[17] = -9.0715e-11
 Dynamics_const_time_7[18]: - 0.037378125865822 u_vars_time_7[10]
   - 0.75 x_vars_time_7[18] + x_vars_time_8[18] = -5.68434e-14
 Dynamics_const_time_7[19]: - 0.018689062932911 u_vars_time_7[10]
   - 0.75 x_vars_time_7[19] + x_vars_time_8[19] = -2.84217e-14
 Dynamics_const_time_7[20]: - 0.018689062932911 u_vars_time_7[10]
   - 0.75 x_vars_time_7[20] + x_vars_time_8[20] = -2.84217e-14
 Dynamics_const_time_7[21]: - 1.4981212771872558e-04 u_vars_time_7[10]
   - 0.75 x_vars_time_7[21] + x_vars_time_8[21] = 2.22045e-16
 Dynamics_const_time_7[22]: - 0.1658677830414493 u_vars_time_7[11]
   - 0.25 x_vars_time_7[18] - 0.25 x_vars_time_7[19]
   - 0.25 x_vars_time_7[20] - x_vars_time_7[22]
   - 0.0758199847444699 x_vars_time_7[23]
   - 0.0521237545883587 x_vars_time_7[24] + x_vars_time_8[22]
   = -1.25056e-12
 Dynamics_const_time_7[23]: u_vars_time_7[12] - 0.000389 x_vars_time_7[15]
   - 0.1945 x_vars_time_7[21] - 0.92372234935164 x_vars_time_7[23]
   + x_vars_time_8[23] = -4.54228e-08
 Dynamics_const_time_7[24]: u_vars_time_7[13] - 0.000111 x_vars_time_7[15]
   - 0.0555 x_vars_time_7[21] - 0.9475616151022549 x_vars_time_7[24]
   + x_vars_time_8[24] = -1.29613e-08
 Dynamics_const_time_7[25]: - u_vars_time_7[12] - u_vars_time_7[13]
   - 0.000457666 x_vars_time_7[23]
   - 3.1463030938647091e-04 x_vars_time_7[24] - x_vars_time_7[25]
   + x_vars_time_8[25] = 5.55112e-17
 All_const_time_7[0]: u_vars_time_7[2] - 0.000389 x_vars_time_7[2]
   - 0.1945 x_vars_time_7[8] <= -0
 All_const_time_7[1]: u_vars_time_7[12] - 0.000389 x_vars_time_7[15]
   - 0.1945 x_vars_time_7[21] <= -0
 All_const_time_7[2]: - u_vars_time_7[3] - u_vars_time_7[13]
   + 0.000111 x_vars_time_7[2] + 0.0555 x_vars_time_7[8]
   + 0.9475616151022548 x_vars_time_7[11] + 0.000111 x_vars_time_7[15]
   + 0.0555 x_vars_time_7[21] + 0.9475616151022548 x_vars_time_7[24]
   <= 2500
 All_const_time_7[3]: u_vars_time_7[3] - 0.000111 x_vars_time_7[2]
   - 0.0555 x_vars_time_7[8] <= -0
 All_const_time_7[4]: u_vars_time_7[13] - 0.000111 x_vars_time_7[15]
   - 0.0555 x_vars_time_7[21] <= -0
 All_const_time_7[5]: u_vars_time_7[0] + u_vars_time_7[10] <= 30000
 All_const_time_7[6]: u_vars_time_7[1] + u_vars_time_7[11] <= 30000
 R59: - u_vars_time_7[8] + 0.25 u_vars_time_7[9] <= -0
 R60: - u_vars_time_7[18] + 0.25 u_vars_time_7[19] <= -0
Bounds
 u_vars_time_7[4] = 1
 u_vars_time_7[5] <= 1
 u_vars_time_7[6] <= 1
 u_vars_time_7[7] <= 1
 u_vars_time_7[8] <= 1
 u_vars_time_7[9] <= 1
 u_vars_time_7[14] = 1
 u_vars_time_7[15] <= 1
 u_vars_time_7[16] <= 1
 u_vars_time_7[17] <= 1
 u_vars_time_7[18] <= 1
 u_vars_time_7[19] <= 1
End
