\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.2307203118011797 u_vars_time_0[0] + 0.793457197386552 u_vars_time_0[1]
   - 333364.241586385 u_vars_time_0[2] - 333488.9330224897 u_vars_time_0[3]
   - 107.9664442847527 u_vars_time_0[4]
   + 187016.1390109332 u_vars_time_0[5] + -0 u_vars_time_0[6]
   + 1.6467075052089248e+07 u_vars_time_0[7]
   + 222905.1586629016 u_vars_time_0[8]
   + 0.4741565494594079 u_vars_time_0[10]
   + 2.321407555176231 u_vars_time_0[11]
   - 438192.9668719928 u_vars_time_0[12]
   - 438356.5464992713 u_vars_time_0[13]
   - 220.8181003529823 u_vars_time_0[14]
   + 187016.1390109332 u_vars_time_0[15] + -0 u_vars_time_0[16]
   + 1.9458369963731349e+07 u_vars_time_0[17]
   + 222905.1586629016 u_vars_time_0[18]
   + 5.8166226640191963e+06 u_vars_time_0[19]
   + 0.1044579302910767 u_vars_time_1[0]
   + 0.6946835135326078 u_vars_time_1[1]
   - 333493.0821925869 u_vars_time_1[2]
   - 333582.1787987602 u_vars_time_1[3] + 1.42109e-13 u_vars_time_1[4]
   + 186822.9818829083 u_vars_time_1[5] + -0 u_vars_time_1[6]
   + 1.6457731861086091e+07 u_vars_time_1[7]
   + 222674.9340389897 u_vars_time_1[8]
   + 0.122637645823083 u_vars_time_1[10]
   + 2.02160009437274 u_vars_time_1[11]
   - 438361.6326656485 u_vars_time_1[12]
   - 438478.634069191 u_vars_time_1[13] + 9.09495e-13 u_vars_time_1[14]
   + 186822.9818829083 u_vars_time_1[15] + -0 u_vars_time_1[16]
   + 1.9438272645509951e+07 u_vars_time_1[17]
   + 222674.9340389897 u_vars_time_1[18]
   + 5.8106150427811146e+06 u_vars_time_1[19]
   + 0.4502396209267461 u_vars_time_2[1]
   - 333633.4095054139 u_vars_time_2[2] - 333681.152808971 u_vars_time_2[3]
   + -0 u_vars_time_2[4] + 186580.3533484999 u_vars_time_2[5]
   + -0 u_vars_time_2[6] + 1.644589533573626e+07 u_vars_time_2[7]
   + 222385.7442811191 u_vars_time_2[8]
   - 0.1041705252828165 u_vars_time_2[10]
   + 1.301824447646295 u_vars_time_2[11]
   - 438545.7281610357 u_vars_time_2[12]
   - 438608.4845314313 u_vars_time_2[13] + -0 u_vars_time_2[14]
   + 186580.3533484999 u_vars_time_2[15] + -0 u_vars_time_2[16]
   + 1.9413028001859166e+07 u_vars_time_2[17]
   + 222385.7442811191 u_vars_time_2[18]
   + 5.803068749505884e+06 u_vars_time_2[19]
   - 333786.1720555538 u_vars_time_3[2]
   - 333786.1720555538 u_vars_time_3[3] + -0 u_vars_time_3[4]
   + 186300.3753232749 u_vars_time_3[5] + -0 u_vars_time_3[6]
   + 1.6432153344210422e+07 u_vars_time_3[7]
   + 222052.037540808 u_vars_time_3[8]
   - 438746.5274871168 u_vars_time_3[12]
   - 438746.5274871168 u_vars_time_3[13] + -0 u_vars_time_3[14]
   + 186300.3753232749 u_vars_time_3[15] + -0 u_vars_time_3[16]
   + 1.9383897275359534e+07 u_vars_time_3[17]
   + 222052.037540808 u_vars_time_3[18]
   + 5.7943608030389948e+06 u_vars_time_3[19]
   + 8.6035500225236285e+06 Constant
Subject To
 Const0_time_0: u_vars_time_0[2] <= 20.30948669370021
 Const1_time_0: u_vars_time_0[12] <= 34.42480936904963
 Const2_time_0: - u_vars_time_0[3] - u_vars_time_0[13]
   <= 2445.736572393941
 Const3_time_0: u_vars_time_0[3] <= 5.795251987148389
 Const4_time_0: u_vars_time_0[13] <= 9.82301758345632
 Const5_time_0: u_vars_time_0[0] + u_vars_time_0[10] <= 30000
 Const6_time_0: u_vars_time_0[1] + u_vars_time_0[11] <= 30000
 R7: - u_vars_time_0[8] + 0.25 u_vars_time_0[9] <= 0
 R8: - u_vars_time_0[18] + 0.25 u_vars_time_0[19] <= 0
 Const0_time_1: u_vars_time_1[2] <= 25.29054794206607
 Const1_time_1: u_vars_time_1[12] <= 42.36445113880747
 Const2_time_1: - 0.9475616151022548 u_vars_time_0[3]
   - 0.9475616151022548 u_vars_time_0[13] - u_vars_time_1[3]
   - u_vars_time_1[13] <= 2429.276904932042
 Const3_time_1: u_vars_time_1[3] <= 7.216583088867182
 Const4_time_1: u_vars_time_1[13] <= 12.08857089050804
 Const5_time_1: u_vars_time_1[0] + u_vars_time_1[10] <= 30000
 Const6_time_1: u_vars_time_1[1] + u_vars_time_1[11] <= 30000
 R16: - u_vars_time_1[8] + 0.25 u_vars_time_1[9] <= 0
 R17: - u_vars_time_1[18] + 0.25 u_vars_time_1[19] <= 0
 Const0_time_2: - 0.5172994782022223 u_vars_time_0[4]
   - 0.1048885422853637 u_vars_time_0[14] + u_vars_time_2[2]
   <= 26.30433964106355
 Const1_time_2: - 0.0484356586497237 u_vars_time_0[4]
   - 0.7266905231405001 u_vars_time_0[14] + u_vars_time_2[12]
   <= 43.92434604419072
 Const2_time_2: - 0.8978730144151937 u_vars_time_0[3]
   + 0.1614308488189358 u_vars_time_0[4]
   - 0.8978730144151937 u_vars_time_0[13]
   + 0.2372886279235755 u_vars_time_0[14]
   - 0.9475616151022548 u_vars_time_1[3]
   - 0.9475616151022548 u_vars_time_1[13] - u_vars_time_2[3]
   - u_vars_time_2[13] <= 2412.945961987318
 Const3_time_2: - 0.1476098768134876 u_vars_time_0[4]
   - 0.0299296354593197 u_vars_time_0[14] + u_vars_time_2[3]
   <= 7.505865553105537
 Const4_time_2: - 0.0138209720054481 u_vars_time_0[4]
   - 0.2073589924642558 u_vars_time_0[14] + u_vars_time_2[13]
   <= 12.53368229024465
 Const5_time_2: u_vars_time_2[0] + u_vars_time_2[10] <= 30000
 Const6_time_2: u_vars_time_2[1] + u_vars_time_2[11] <= 30000
 R25: - u_vars_time_2[8] + 0.25 u_vars_time_2[9] <= 0
 R26: - u_vars_time_2[18] + 0.25 u_vars_time_2[19] <= 0
 Const0_time_3: 1.23012e-07 u_vars_time_0[0]
   - 0.774009702486485 u_vars_time_0[4] + 1.25887e-07 u_vars_time_0[10]
   - 0.1569395540290856 u_vars_time_0[14]
   - 0.634506576009259 u_vars_time_1[4]
   - 0.1277456966757103 u_vars_time_1[14] + u_vars_time_3[2]
   <= 24.98255839880488
 Const1_time_3: 5.80471e-08 u_vars_time_0[0]
   - 0.0723943194789755 u_vars_time_0[4] + 2.51635e-07 u_vars_time_0[10]
   - 1.086147424463223 u_vars_time_0[14]
   - 0.0596169890753234 u_vars_time_1[4]
   - 0.8852874707711347 u_vars_time_1[14] + u_vars_time_3[12]
   <= 41.75033990688697
 Const2_time_3: - 5.16645e-08 u_vars_time_0[0]
   - 0.8507900036959912 u_vars_time_0[3]
   + 0.3944845612793558 u_vars_time_0[4] - 1.07725e-07 u_vars_time_0[10]
   - 0.8507900036959912 u_vars_time_0[13]
   + 0.5795567898976438 u_vars_time_0[14]
   - 0.8978730144151937 u_vars_time_1[3]
   + 0.1980661072606392 u_vars_time_1[4]
   - 0.8978730144151937 u_vars_time_1[13]
   + 0.2890660195028272 u_vars_time_1[14]
   - 0.9475616151022548 u_vars_time_2[3]
   - 0.9475616151022548 u_vars_time_2[13] - u_vars_time_3[3]
   - u_vars_time_3[13] <= 2398.468899909715
 Const3_time_3: 3.5101e-08 u_vars_time_0[0]
   - 0.2208613804010278 u_vars_time_0[4] + 3.59214e-08 u_vars_time_0[10]
   - 0.0447822377820784 u_vars_time_0[14]
   - 0.1810545756735932 u_vars_time_1[4]
   - 0.0364518568920407 u_vars_time_1[14] + u_vars_time_3[3]
   <= 7.128699183206533
 Const4_time_3: 1.65636e-08 u_vars_time_0[0]
   - 0.0206575050441292 u_vars_time_0[4] + 7.18033e-08 u_vars_time_0[10]
   - 0.3099289565949042 u_vars_time_0[14]
   - 0.017011531587046 u_vars_time_1[4]
   - 0.2526141626107865 u_vars_time_1[14] + u_vars_time_3[13]
   <= 11.91333606597546
 Const5_time_3: u_vars_time_3[0] + u_vars_time_3[10] <= 30000
 Const6_time_3: u_vars_time_3[1] + u_vars_time_3[11] <= 30000
 R34: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= 0
 R35: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= 0
Bounds
 u_vars_time_0[4] = 1
 u_vars_time_0[5] <= 1
 u_vars_time_0[6] <= 1
 u_vars_time_0[7] <= 1
 u_vars_time_0[8] <= 1
 u_vars_time_0[9] <= 1
 u_vars_time_0[14] = 1
 u_vars_time_0[15] <= 1
 u_vars_time_0[16] <= 1
 u_vars_time_0[17] <= 1
 u_vars_time_0[18] <= 1
 u_vars_time_0[19] <= 1
 u_vars_time_1[4] = 1
 u_vars_time_1[5] <= 1
 u_vars_time_1[6] <= 1
 u_vars_time_1[7] <= 1
 u_vars_time_1[8] <= 1
 u_vars_time_1[9] <= 1
 u_vars_time_1[14] = 1
 u_vars_time_1[15] <= 1
 u_vars_time_1[16] <= 1
 u_vars_time_1[17] <= 1
 u_vars_time_1[18] <= 1
 u_vars_time_1[19] <= 1
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
