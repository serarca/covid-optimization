\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  177610.5543185544 u_vars_time_23[5] + -0 u_vars_time_23[6]
   + 1.6233874207700251e+07 u_vars_time_23[7]
   + 211694.610956912 u_vars_time_23[8]
   + 177610.5543185544 u_vars_time_23[15] + -0 u_vars_time_23[16]
   + 1.8479752034618743e+07 u_vars_time_23[17]
   + 211694.610956912 u_vars_time_23[18]
   + 5.5240878198106755e+06 u_vars_time_23[19]
   + 10.32536320952022 x_vars_time_23[0]
   + 10.32536320952022 x_vars_time_23[1]
   + 10.32536320952022 x_vars_time_23[2]
   + 10.32536320952022 x_vars_time_23[3]
   + 10.32536320952022 x_vars_time_23[9]
   + 18.29768039680393 x_vars_time_23[13]
   + 18.29768039680393 x_vars_time_23[14]
   + 18.29768039680393 x_vars_time_23[15]
   + 18.29768039680393 x_vars_time_23[16]
   + 18.29768039680393 x_vars_time_23[22]
   - 333786.1720555538 x_vars_time_24[12]
   - 438746.5274871168 x_vars_time_24[25]
Subject To
 Initial_Conditions: x_vars_time_23[0] = 1.0692212564485248e+06
 Initial_Conditions: x_vars_time_23[1] = 47157.7228321989
 Initial_Conditions: x_vars_time_23[2] = 54002.62530512802
 Initial_Conditions: x_vars_time_23[3] = 401851.1467746288
 Initial_Conditions: x_vars_time_23[4] = 1.5722327513604807e+06
 Initial_Conditions: x_vars_time_23[5] = -0
 Initial_Conditions: x_vars_time_23[6] = -0
 Initial_Conditions: x_vars_time_23[7] = -0
 Initial_Conditions: x_vars_time_23[8] = -0
 Initial_Conditions: x_vars_time_23[9] = 22240.07753568498
 Initial_Conditions: x_vars_time_23[10] = 278.2411753230149
 Initial_Conditions: x_vars_time_23[11] = 102.3874514858729
 Initial_Conditions: x_vars_time_23[12] = 2.542472458970185
 Initial_Conditions: x_vars_time_23[13] = 545392.2814749761
 Initial_Conditions: x_vars_time_23[14] = 79925.86190608045
 Initial_Conditions: x_vars_time_23[15] = 101340.7319185389
 Initial_Conditions: x_vars_time_23[16] = 741629.1869447615
 Initial_Conditions: x_vars_time_23[17] = 1.4682880622443575e+06
 Initial_Conditions: x_vars_time_23[18] = 1.528515872277747
 Initial_Conditions: x_vars_time_23[19] = 0.7642579361388735
 Initial_Conditions: x_vars_time_23[20] = 0.7642579361388735
 Initial_Conditions: x_vars_time_23[21] = 0.0061263161213537
 Initial_Conditions: x_vars_time_23[22] = 77057.22822340352
 Initial_Conditions: x_vars_time_23[23] = 553.8925229879833
 Initial_Conditions: x_vars_time_23[24] = 201.9194096676336
 Initial_Conditions: x_vars_time_23[25] = 4.834433432430517
 Dynamics_const_time_23[0]: 4392.387426800735 u_vars_time_23[4]
   + 3615.265429410076 u_vars_time_23[5]
   + 4614.829731254808 u_vars_time_23[7]
   + 272.6522016608907 u_vars_time_23[8]
   + 945.0922243746267 u_vars_time_23[14]
   + 602.617776529343 u_vars_time_23[15]
   + 333.2893622316585 u_vars_time_23[17]
   + 73.20543833797247 u_vars_time_23[18]
   - 0.9907413365873883 x_vars_time_23[0]
   + 0.1109418308881851 x_vars_time_23[2]
   - 0.00375744886682524 x_vars_time_23[4]
   - 0.00375744886682524 x_vars_time_23[9]
   + 0.0385670157394256 x_vars_time_23[15]
   - 0.0025291497162839 x_vars_time_23[17]
   - 0.0025291497162839 x_vars_time_23[22] + x_vars_time_24[0]
   = 14849.33959060019
 Dynamics_const_time_23[1]: - 4392.387426800735 u_vars_time_23[4]
   - 3615.265429410076 u_vars_time_23[5]
   - 4614.829731254808 u_vars_time_23[7]
   - 272.6522016608907 u_vars_time_23[8]
   - 945.0922243746267 u_vars_time_23[14]
   - 602.617776529343 u_vars_time_23[15]
   - 333.2893622316585 u_vars_time_23[17]
   - 73.20543833797247 u_vars_time_23[18]
   - 0.00925866341261177 x_vars_time_23[0] - 0.75 x_vars_time_23[1]
   - 0.1109418308881851 x_vars_time_23[2]
   + 0.00375744886682524 x_vars_time_23[4]
   + 0.00375744886682524 x_vars_time_23[9]
   - 0.0385670157394256 x_vars_time_23[15]
   + 0.0025291497162839 x_vars_time_23[17]
   + 0.0025291497162839 x_vars_time_23[22] + x_vars_time_24[1]
   = -14849.33959060011
 Dynamics_const_time_23[2]: 0.0343477295320293 u_vars_time_23[0]
   - 0.25 x_vars_time_23[1] - 0.75 x_vars_time_23[2] + x_vars_time_24[2]
   = -0
 Dynamics_const_time_23[3]: 0.2555926572747578 u_vars_time_23[1]
   - 0.2495 x_vars_time_23[2] - x_vars_time_23[3] + x_vars_time_24[3]
   = 2.72848e-11
 Dynamics_const_time_23[4]: 0.0343477295320293 u_vars_time_23[0]
   + 0.2555926572747578 u_vars_time_23[1] + 0.0005 x_vars_time_23[2]
   - x_vars_time_23[4] + x_vars_time_24[4] = -3.87814e-11
 Dynamics_const_time_23[5]: - 0.0171395170364826 u_vars_time_23[0]
   - 0.75 x_vars_time_23[5] + x_vars_time_24[5] = -0
 Dynamics_const_time_23[6]: - 0.0085697585182413 u_vars_time_23[0]
   - 0.75 x_vars_time_23[6] + x_vars_time_24[6] = -0
 Dynamics_const_time_23[7]: - 0.0085697585182413 u_vars_time_23[0]
   - 0.75 x_vars_time_23[7] + x_vars_time_24[7] = -0
 Dynamics_const_time_23[8]: - 6.86955e-05 u_vars_time_23[0]
   - 0.75 x_vars_time_23[8] + x_vars_time_24[8] = -0
 Dynamics_const_time_23[9]: - 0.2555926572747578 u_vars_time_23[1]
   - 0.25 x_vars_time_23[5] - 0.25 x_vars_time_23[6]
   - 0.25 x_vars_time_23[7] - x_vars_time_23[9]
   - 0.0758199847444699 x_vars_time_23[10]
   - 0.0521237545883587 x_vars_time_23[11] + x_vars_time_24[9]
   = 1.09424e-12
 Dynamics_const_time_23[10]: u_vars_time_23[2] - 0.000389 x_vars_time_23[2]
   - 0.1945 x_vars_time_23[8] - 0.92372234935164 x_vars_time_23[10]
   + x_vars_time_24[10] = -2.1007e-08
 Dynamics_const_time_23[11]: u_vars_time_23[3] - 0.000111 x_vars_time_23[2]
   - 0.0555 x_vars_time_23[8] - 0.9475616151022549 x_vars_time_23[11]
   + x_vars_time_24[11] = -5.99429e-09
 Dynamics_const_time_23[12]: - u_vars_time_23[2] - u_vars_time_23[3]
   - 0.000457666 x_vars_time_23[10]
   - 3.1463030938647091e-04 x_vars_time_23[11] - x_vars_time_23[12]
   + x_vars_time_24[12] = -1.11022e-16
 Dynamics_const_time_23[13]: 231.1288384301203 u_vars_time_23[4]
   + 147.3743455826088 u_vars_time_23[5]
   + 81.50821781463024 u_vars_time_23[7]
   + 17.90289605502981 u_vars_time_23[8]
   + 3790.42518879484 u_vars_time_23[14]
   + 6933.983360372074 u_vars_time_23[15]
   + 9591.382571036942 u_vars_time_23[17]
   + 1220.080101700664 u_vars_time_23[18]
   + 16.77785050965946 u_vars_time_23[19]
   + 0.0176996690506084 x_vars_time_23[2]
   - 5.9946370890903602e-04 x_vars_time_23[4]
   - 5.9946370890903602e-04 x_vars_time_23[9]
   - 0.9730706818821413 x_vars_time_23[13]
   + 0.1354955050269147 x_vars_time_23[15]
   - 0.00888553110803042 x_vars_time_23[17]
   - 0.00888553110803042 x_vars_time_23[22] + x_vars_time_24[13]
   = 22030.56337029654
 Dynamics_const_time_23[14]: - 231.1288384301203 u_vars_time_23[4]
   - 147.3743455826088 u_vars_time_23[5]
   - 81.50821781463024 u_vars_time_23[7]
   - 17.90289605502981 u_vars_time_23[8]
   - 3790.42518879484 u_vars_time_23[14]
   - 6933.983360372074 u_vars_time_23[15]
   - 9591.382571036942 u_vars_time_23[17]
   - 1220.080101700664 u_vars_time_23[18]
   - 16.77785050965946 u_vars_time_23[19]
   - 0.0176996690506084 x_vars_time_23[2]
   + 5.9946370890903602e-04 x_vars_time_23[4]
   + 5.9946370890903602e-04 x_vars_time_23[9]
   - 0.0269293181178587 x_vars_time_23[13] - 0.75 x_vars_time_23[14]
   - 0.1354955050269147 x_vars_time_23[15]
   + 0.00888553110803042 x_vars_time_23[17]
   + 0.00888553110803042 x_vars_time_23[22] + x_vars_time_24[14]
   = -22030.56337029656
 Dynamics_const_time_23[15]: 0.0690196525630223 u_vars_time_23[10]
   - 0.25 x_vars_time_23[14] - 0.75 x_vars_time_23[15] + x_vars_time_24[15]
   = -0
 Dynamics_const_time_23[16]: 0.505097879643005 u_vars_time_23[11]
   - 0.2495 x_vars_time_23[15] - x_vars_time_23[16] + x_vars_time_24[16]
   = -3.63798e-12
 Dynamics_const_time_23[17]: 0.0690196525630223 u_vars_time_23[10]
   + 0.505097879643005 u_vars_time_23[11] + 0.0005 x_vars_time_23[15]
   - x_vars_time_23[17] + x_vars_time_24[17] = 9.63638e-11
 Dynamics_const_time_23[18]: - 0.0344408066289481 u_vars_time_23[10]
   - 0.75 x_vars_time_23[18] + x_vars_time_24[18] = -1.11022e-16
 Dynamics_const_time_23[19]: - 0.0172204033144741 u_vars_time_23[10]
   - 0.75 x_vars_time_23[19] + x_vars_time_24[19] = -5.55112e-17
 Dynamics_const_time_23[20]: - 0.0172204033144741 u_vars_time_23[10]
   - 0.75 x_vars_time_23[20] + x_vars_time_24[20] = -5.55112e-17
 Dynamics_const_time_23[21]: - 1.3803930512604467e-04 u_vars_time_23[10]
   - 0.75 x_vars_time_23[21] + x_vars_time_24[21] = -0
 Dynamics_const_time_23[22]: - 0.505097879643005 u_vars_time_23[11]
   - 0.25 x_vars_time_23[18] - 0.25 x_vars_time_23[19]
   - 0.25 x_vars_time_23[20] - x_vars_time_23[22]
   - 0.0758199847444699 x_vars_time_23[23]
   - 0.0521237545883587 x_vars_time_23[24] + x_vars_time_24[22]
   = 5.08749e-12
 Dynamics_const_time_23[23]: u_vars_time_23[12]
   - 0.000389 x_vars_time_23[15] - 0.1945 x_vars_time_23[21]
   - 0.92372234935164 x_vars_time_23[23] + x_vars_time_24[23]
   = -3.94228e-08
 Dynamics_const_time_23[24]: u_vars_time_23[13]
   - 0.000111 x_vars_time_23[15] - 0.0555 x_vars_time_23[21]
   - 0.9475616151022549 x_vars_time_23[24] + x_vars_time_24[24]
   = -1.12492e-08
 Dynamics_const_time_23[25]: - u_vars_time_23[12] - u_vars_time_23[13]
   - 0.000457666 x_vars_time_23[23]
   - 3.1463030938647091e-04 x_vars_time_23[24] - x_vars_time_23[25]
   + x_vars_time_24[25] = 3.88578e-16
 All_const_time_23[0]: u_vars_time_23[2] - 0.000389 x_vars_time_23[2]
   - 0.1945 x_vars_time_23[8] <= -0
 All_const_time_23[1]: u_vars_time_23[12] - 0.000389 x_vars_time_23[15]
   - 0.1945 x_vars_time_23[21] <= -0
 All_const_time_23[2]: - u_vars_time_23[3] - u_vars_time_23[13]
   + 0.000111 x_vars_time_23[2] + 0.0555 x_vars_time_23[8]
   + 0.9475616151022548 x_vars_time_23[11] + 0.000111 x_vars_time_23[15]
   + 0.0555 x_vars_time_23[21] + 0.9475616151022548 x_vars_time_23[24]
   <= 2500
 All_const_time_23[3]: u_vars_time_23[3] - 0.000111 x_vars_time_23[2]
   - 0.0555 x_vars_time_23[8] <= -0
 All_const_time_23[4]: u_vars_time_23[13] - 0.000111 x_vars_time_23[15]
   - 0.0555 x_vars_time_23[21] <= -0
 All_const_time_23[5]: u_vars_time_23[0] + u_vars_time_23[10] <= 30000
 All_const_time_23[6]: u_vars_time_23[1] + u_vars_time_23[11] <= 30000
 R59: - u_vars_time_23[8] + 0.25 u_vars_time_23[9] <= -0
 R60: - u_vars_time_23[18] + 0.25 u_vars_time_23[19] <= -0
Bounds
 u_vars_time_23[4] = 1
 u_vars_time_23[5] <= 1
 u_vars_time_23[6] <= 1
 u_vars_time_23[7] <= 1
 u_vars_time_23[8] <= 1
 u_vars_time_23[9] <= 1
 u_vars_time_23[14] = 1
 u_vars_time_23[15] <= 1
 u_vars_time_23[16] <= 1
 u_vars_time_23[17] <= 1
 u_vars_time_23[18] <= 1
 u_vars_time_23[19] <= 1
End
