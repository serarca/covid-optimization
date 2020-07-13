\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  186411.794809836 u_vars_time_2[5] + -0 u_vars_time_2[6]
   + 1.6466469863246087e+07 u_vars_time_2[7]
   + 222184.8387977551 u_vars_time_2[8]
   + 186411.794809836 u_vars_time_2[15] + -0 u_vars_time_2[16]
   + 1.9395490080141746e+07 u_vars_time_2[17]
   + 222184.8387977551 u_vars_time_2[18]
   + 5.797826199737763e+06 u_vars_time_2[19]
   + 10.32536320952022 x_vars_time_2[0]
   + 10.32536320952022 x_vars_time_2[1]
   + 10.32536320952022 x_vars_time_2[2]
   + 10.32536320952022 x_vars_time_2[3]
   + 10.32536320952022 x_vars_time_2[9]
   + 18.29768039680393 x_vars_time_2[13]
   + 18.29768039680393 x_vars_time_2[14]
   + 18.29768039680393 x_vars_time_2[15]
   + 18.29768039680393 x_vars_time_2[16]
   + 18.29768039680393 x_vars_time_2[22] + -0 u_vars_time_3[5]
   + -0 u_vars_time_3[6] + -0 u_vars_time_3[7] + -0 u_vars_time_3[15]
   + -0 u_vars_time_3[16] + -0 u_vars_time_3[17]
   - 1.1159708999999999e+06 x_vars_time_4[12]
   - 1.1159708999999999e+06 x_vars_time_4[25]
Subject To
 Initial_Conditions: x_vars_time_2[0] = 1.3712161275384775e+06
 Initial_Conditions: x_vars_time_2[1] = 81911.73194382116
 Initial_Conditions: x_vars_time_2[2] = 71161.48166790156
 Initial_Conditions: x_vars_time_2[3] = 70470.04204163088
 Initial_Conditions: x_vars_time_2[4] = 1.5947593831918309e+06
 Initial_Conditions: x_vars_time_2[5] = -0
 Initial_Conditions: x_vars_time_2[6] = -0
 Initial_Conditions: x_vars_time_2[7] = -0
 Initial_Conditions: x_vars_time_2[8] = -0
 Initial_Conditions: x_vars_time_2[9] = 13.47234745651504
 Initial_Conditions: x_vars_time_2[10] = 56.94941171509234
 Initial_Conditions: x_vars_time_2[11] = 26.11407467156168
 Initial_Conditions: x_vars_time_2[12] = 0.080970466978301
 Initial_Conditions: x_vars_time_2[13] = 1.1529669084935721e+06
 Initial_Conditions: x_vars_time_2[14] = 151424.4315921173
 Initial_Conditions: x_vars_time_2[15] = 119487.1819578096
 Initial_Conditions: x_vars_time_2[16] = 117168.4840365533
 Initial_Conditions: x_vars_time_2[17] = 1.5410470060800523e+06
 Initial_Conditions: x_vars_time_2[18] = 642.6631843856138
 Initial_Conditions: x_vars_time_2[19] = 321.3315921928069
 Initial_Conditions: x_vars_time_2[20] = 321.3315921928069
 Initial_Conditions: x_vars_time_2[21] = 2.575804346234925
 Initial_Conditions: x_vars_time_2[22] = 3632.858786315811
 Initial_Conditions: x_vars_time_2[23] = 96.48323577291063
 Initial_Conditions: x_vars_time_2[24] = 44.60902037709108
 Initial_Conditions: x_vars_time_2[25] = 0.1406976707387382
 Dynamics_const_time_2[0]: 7087.096518897456 u_vars_time_2[4]
   + 5895.214136380773 u_vars_time_2[5] + 7679.36479261767 u_vars_time_2[7]
   + 434.7794867461091 u_vars_time_2[8]
   + 1429.673436674557 u_vars_time_2[14]
   + 911.6005881245893 u_vars_time_2[15]
   + 504.1782543752344 u_vars_time_2[17]
   + 110.7403784653615 u_vars_time_2[18]
   - 0.9883059292602985 x_vars_time_2[0]
   + 0.1422498919684666 x_vars_time_2[2]
   - 0.00634743251643315 x_vars_time_2[4]
   - 0.00634743251643315 x_vars_time_2[9]
   + 0.0494813353064693 x_vars_time_2[15]
   - 0.00382757971393054 x_vars_time_2[17]
   - 0.00382757971393054 x_vars_time_2[22] + x_vars_time_3[0]
   = 24052.64759228179
 Dynamics_const_time_2[1]: - 7087.096518897456 u_vars_time_2[4]
   - 5895.214136380773 u_vars_time_2[5] - 7679.36479261767 u_vars_time_2[7]
   - 434.7794867461091 u_vars_time_2[8]
   - 1429.673436674557 u_vars_time_2[14]
   - 911.6005881245893 u_vars_time_2[15]
   - 504.1782543752344 u_vars_time_2[17]
   - 110.7403784653615 u_vars_time_2[18]
   - 0.0116940707397015 x_vars_time_2[0] - 0.75 x_vars_time_2[1]
   - 0.1422498919684666 x_vars_time_2[2]
   + 0.00634743251643315 x_vars_time_2[4]
   + 0.00634743251643315 x_vars_time_2[9]
   - 0.0494813353064693 x_vars_time_2[15]
   + 0.00382757971393054 x_vars_time_2[17]
   + 0.00382757971393054 x_vars_time_2[22] + x_vars_time_3[1]
   = -24052.64759228176
 Dynamics_const_time_2[2]: 0.0446220805583068 u_vars_time_2[0]
   - 0.25 x_vars_time_2[1] - 0.75 x_vars_time_2[2] + x_vars_time_3[2]
   = -7.27596e-12
 Dynamics_const_time_2[3]: 0.0441885106834039 u_vars_time_2[1]
   - 0.2495 x_vars_time_2[2] - x_vars_time_2[3] + x_vars_time_3[3]
   = 3.63798e-12
 Dynamics_const_time_2[4]: 0.0446220805583068 u_vars_time_2[0]
   + 0.0441885106834039 u_vars_time_2[1] + 0.0005 x_vars_time_2[2]
   - x_vars_time_2[4] + x_vars_time_3[4] = 6.29541e-11
 Dynamics_const_time_2[5]: - 0.0222664181985951 u_vars_time_2[0]
   - 0.75 x_vars_time_2[5] + x_vars_time_3[5] = -0
 Dynamics_const_time_2[6]: - 0.0111332090992976 u_vars_time_2[0]
   - 0.75 x_vars_time_2[6] + x_vars_time_3[6] = -0
 Dynamics_const_time_2[7]: - 0.0111332090992976 u_vars_time_2[0]
   - 0.75 x_vars_time_2[7] + x_vars_time_3[7] = -0
 Dynamics_const_time_2[8]: - 8.92442e-05 u_vars_time_2[0]
   - 0.75 x_vars_time_2[8] + x_vars_time_3[8] = -0
 Dynamics_const_time_2[9]: - 0.0441885106834039 u_vars_time_2[1]
   - 0.25 x_vars_time_2[5] - 0.25 x_vars_time_2[6] - 0.25 x_vars_time_2[7]
   - x_vars_time_2[9] - 0.0758199847444699 x_vars_time_2[10]
   - 0.0521237545883587 x_vars_time_2[11] + x_vars_time_3[9]
   = -1.77636e-15
 Dynamics_const_time_2[10]: u_vars_time_2[2] - 0.000389 x_vars_time_2[2]
   - 0.1945 x_vars_time_2[8] - 0.92372234935164 x_vars_time_2[10]
   + x_vars_time_3[10] = -2.76818e-08
 Dynamics_const_time_2[11]: u_vars_time_2[3] - 0.000111 x_vars_time_2[2]
   - 0.0555 x_vars_time_2[8] - 0.9475616151022549 x_vars_time_2[11]
   + x_vars_time_3[11] = -7.89893e-09
 Dynamics_const_time_2[12]: - u_vars_time_2[2] - u_vars_time_2[3]
   - 0.000457666 x_vars_time_2[10]
   - 3.1463030938647091e-04 x_vars_time_2[11] - x_vars_time_2[12]
   + x_vars_time_3[12] = -0
 Dynamics_const_time_2[13]: 643.739841402939 u_vars_time_2[4]
   + 410.4669001782476 u_vars_time_2[5]
   + 227.0166179409504 u_vars_time_2[7]
   + 49.86313064781233 u_vars_time_2[8]
   + 9586.701296877918 u_vars_time_2[14]
   + 17376.73505434721 u_vars_time_2[15]
   + 23964.89912101765 u_vars_time_2[17]
   + 3052.869368395855 u_vars_time_2[18]
   + 41.83776780709456 u_vars_time_2[19]
   + 0.0374103084694583 x_vars_time_2[2]
   - 0.00166931169607829 x_vars_time_2[4]
   - 0.00166931169607829 x_vars_time_2[9]
   - 0.9679932247312923 x_vars_time_2[13]
   + 0.2865627859912266 x_vars_time_2[15]
   - 0.0221667806584847 x_vars_time_2[17]
   - 0.0221667806584847 x_vars_time_2[22] + x_vars_time_3[13]
   = 55354.12909861555
 Dynamics_const_time_2[14]: - 643.739841402939 u_vars_time_2[4]
   - 410.4669001782476 u_vars_time_2[5]
   - 227.0166179409504 u_vars_time_2[7]
   - 49.86313064781233 u_vars_time_2[8]
   - 9586.701296877918 u_vars_time_2[14]
   - 17376.73505434721 u_vars_time_2[15]
   - 23964.89912101765 u_vars_time_2[17]
   - 3052.869368395855 u_vars_time_2[18]
   - 41.83776780709456 u_vars_time_2[19]
   - 0.0374103084694583 x_vars_time_2[2]
   + 0.00166931169607829 x_vars_time_2[4]
   + 0.00166931169607829 x_vars_time_2[9]
   - 0.0320067752687077 x_vars_time_2[13] - 0.75 x_vars_time_2[14]
   - 0.2865627859912266 x_vars_time_2[15]
   + 0.0221667806584847 x_vars_time_2[17]
   + 0.0221667806584847 x_vars_time_2[22] + x_vars_time_3[14]
   = -55354.12909861565
 Dynamics_const_time_2[15]: 0.0775363642292444 u_vars_time_2[10]
   - 0.25 x_vars_time_2[14] - 0.7305327158213617 x_vars_time_2[15]
   - 0.0015094224366291 x_vars_time_2[17] + x_vars_time_3[15]
   = 4.54747e-12
 Dynamics_const_time_2[16]: 0.0760317391839939 u_vars_time_2[11]
   - 0.2495 x_vars_time_2[15] - x_vars_time_2[16] + x_vars_time_3[16] = -0
 Dynamics_const_time_2[17]: 0.0775363642292444 u_vars_time_2[10]
   + 0.0760317391839939 u_vars_time_2[11]
   + 0.0199672841786382 x_vars_time_2[15]
   - 1.001509422436629 x_vars_time_2[17] + x_vars_time_3[17] = 1.09594e-10
 Dynamics_const_time_2[18]: - 0.038690645750393 u_vars_time_2[10]
   - 0.00971417480514047 x_vars_time_2[15]
   + 7.5320179587792118e-04 x_vars_time_2[17] - 0.75 x_vars_time_2[18]
   + x_vars_time_3[18] = -4.54747e-13
 Dynamics_const_time_2[19]: - 0.0193453228751965 u_vars_time_2[10]
   - 0.00485708740257024 x_vars_time_2[15]
   + 3.7660089793896059e-04 x_vars_time_2[17] - 0.75 x_vars_time_2[19]
   + x_vars_time_3[19] = -1.13687e-13
 Dynamics_const_time_2[20]: - 0.0193453228751965 u_vars_time_2[10]
   - 0.00485708740257024 x_vars_time_2[15]
   + 3.7660089793896059e-04 x_vars_time_2[17] - 0.75 x_vars_time_2[20]
   + x_vars_time_3[20] = -2.27374e-13
 Dynamics_const_time_2[21]: - 1.5507272845848889e-04 u_vars_time_2[10]
   - 3.89346e-05 x_vars_time_2[15] + 3.01884e-06 x_vars_time_2[17]
   - 0.75 x_vars_time_2[21] + x_vars_time_3[21] = -1.77636e-15
 Dynamics_const_time_2[22]: - 0.0760317391839939 u_vars_time_2[11]
   - 0.25 x_vars_time_2[18] - 0.25 x_vars_time_2[19]
   - 0.25 x_vars_time_2[20] - x_vars_time_2[22]
   - 0.0758199847444699 x_vars_time_2[23]
   - 0.0521237545883587 x_vars_time_2[24] + x_vars_time_3[22] = -0
 Dynamics_const_time_2[23]: u_vars_time_2[12] - 0.000389 x_vars_time_2[15]
   - 0.1945 x_vars_time_2[21] - 0.92372234935164 x_vars_time_2[23]
   + x_vars_time_3[23] = -4.69815e-08
 Dynamics_const_time_2[24]: u_vars_time_2[13] - 0.000111 x_vars_time_2[15]
   - 0.0555 x_vars_time_2[21] - 0.9475616151022549 x_vars_time_2[24]
   + x_vars_time_3[24] = -1.3406e-08
 Dynamics_const_time_2[25]: - u_vars_time_2[12] - u_vars_time_2[13]
   - 0.000457666 x_vars_time_2[23]
   - 3.1463030938647091e-04 x_vars_time_2[24] - x_vars_time_2[25]
   + x_vars_time_3[25] = -0
 All_const_time_2[0]: u_vars_time_2[2] - 0.000389 x_vars_time_2[2]
   - 0.1945 x_vars_time_2[8] <= -0
 All_const_time_2[1]: u_vars_time_2[12] - 0.000389 x_vars_time_2[15]
   - 0.1945 x_vars_time_2[21] <= -0
 All_const_time_2[2]: - u_vars_time_2[3] - u_vars_time_2[13]
   + 0.000111 x_vars_time_2[2] + 0.0555 x_vars_time_2[8]
   + 0.9475616151022548 x_vars_time_2[11] + 0.000111 x_vars_time_2[15]
   + 0.0555 x_vars_time_2[21] + 0.9475616151022548 x_vars_time_2[24]
   <= 2500
 All_const_time_2[3]: u_vars_time_2[3] - 0.000111 x_vars_time_2[2]
   - 0.0555 x_vars_time_2[8] <= -0
 All_const_time_2[4]: u_vars_time_2[13] - 0.000111 x_vars_time_2[15]
   - 0.0555 x_vars_time_2[21] <= -0
 All_const_time_2[5]: u_vars_time_2[0] + u_vars_time_2[10] <= 30000
 All_const_time_2[6]: u_vars_time_2[1] + u_vars_time_2[11] <= 30000
 R59: - u_vars_time_2[8] + 0.25 u_vars_time_2[9] <= -0
 R60: - u_vars_time_2[18] + 0.25 u_vars_time_2[19] <= -0
 Dynamics_const_time_3[0]: 7299.864695603907 u_vars_time_3[4]
   + 1481.835355372001 u_vars_time_3[14]
   - 0.9956799375325154 x_vars_time_3[0]
   + 0.0391446635468153 x_vars_time_3[2]
   - 0.00181270340959928 x_vars_time_3[4]
   - 0.00181270340959928 x_vars_time_3[9]
   + 0.0236818183432245 x_vars_time_3[15]
   - 0.00192118674724218 x_vars_time_3[17]
   - 0.00192118674724218 x_vars_time_3[22] + x_vars_time_4[0]
   = 8781.700050975995
 Dynamics_const_time_3[1]: - 7299.864695603907 u_vars_time_3[4]
   - 1481.835355372001 u_vars_time_3[14]
   - 0.00432006246748466 x_vars_time_3[0] - 0.75 x_vars_time_3[1]
   - 0.0391446635468153 x_vars_time_3[2]
   + 0.00181270340959928 x_vars_time_3[4]
   + 0.00181270340959928 x_vars_time_3[9]
   - 0.0236818183432245 x_vars_time_3[15]
   + 0.00192118674724218 x_vars_time_3[17]
   + 0.00192118674724218 x_vars_time_3[22] + x_vars_time_4[1]
   = -8781.700050975913
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
   + 9718.586496681311 u_vars_time_3[14]
   + 0.0175136175991157 x_vars_time_3[2]
   - 8.1101717219684156e-04 x_vars_time_3[4]
   - 8.1101717219684156e-04 x_vars_time_3[9]
   - 0.9938084392947079 x_vars_time_3[13]
   + 0.0448823120641894 x_vars_time_3[15]
   - 0.00364107611474769 x_vars_time_3[17]
   - 0.00364107611474769 x_vars_time_3[22] + x_vars_time_4[13]
   = 10365.26845709379
 Dynamics_const_time_3[14]: - 646.6819604124588 u_vars_time_3[4]
   - 9718.586496681311 u_vars_time_3[14]
   - 0.0175136175991157 x_vars_time_3[2]
   + 8.1101717219684156e-04 x_vars_time_3[4]
   + 8.1101717219684156e-04 x_vars_time_3[9]
   - 0.00619156070529213 x_vars_time_3[13] - 0.75 x_vars_time_3[14]
   - 0.0448823120641894 x_vars_time_3[15]
   + 0.00364107611474769 x_vars_time_3[17]
   + 0.00364107611474769 x_vars_time_3[22] + x_vars_time_4[14]
   = -10365.26845709377
 Dynamics_const_time_3[15]: 0.0813339582180069 u_vars_time_3[10]
   - 0.25 x_vars_time_3[14] - 0.75 x_vars_time_3[15] + x_vars_time_4[15]
   = 7.27596e-12
 Dynamics_const_time_3[16]: 0.0955249528951186 u_vars_time_3[11]
   - 0.2495 x_vars_time_3[15] - x_vars_time_3[16] + x_vars_time_4[16]
   = -3.63798e-12
 Dynamics_const_time_3[17]: 0.0813339582180069 u_vars_time_3[10]
   + 0.0955249528951186 u_vars_time_3[11] + 0.0005 x_vars_time_3[15]
   - x_vars_time_3[17] + x_vars_time_4[17] = -6.77645e-11
 Dynamics_const_time_3[18]: - 0.0405856451507854 u_vars_time_3[10]
   - 0.75 x_vars_time_3[18] + x_vars_time_4[18] = -0
 Dynamics_const_time_3[19]: - 0.0202928225753927 u_vars_time_3[10]
   - 0.75 x_vars_time_3[19] + x_vars_time_4[19] = -0
 Dynamics_const_time_3[20]: - 0.0202928225753927 u_vars_time_3[10]
   - 0.75 x_vars_time_3[20] + x_vars_time_4[20] = -0
 Dynamics_const_time_3[21]: - 0.000162668 u_vars_time_3[10]
   - 0.75 x_vars_time_3[21] + x_vars_time_4[21] = 4.44089e-16
 Dynamics_const_time_3[22]: - 0.0955249528951186 u_vars_time_3[11]
   - 0.25 x_vars_time_3[18] - 0.25 x_vars_time_3[19]
   - 0.25 x_vars_time_3[20] - x_vars_time_3[22]
   - 0.0758199847444699 x_vars_time_3[23]
   - 0.0521237545883587 x_vars_time_3[24] + x_vars_time_4[22] = -0
 Dynamics_const_time_3[23]: u_vars_time_3[12] - 0.000389 x_vars_time_3[15]
   - 0.1945 x_vars_time_3[21] - 0.92372234935164 x_vars_time_3[23]
   + x_vars_time_4[23] = -4.99621e-08
 Dynamics_const_time_3[24]: u_vars_time_3[13] - 0.000111 x_vars_time_3[15]
   - 0.0555 x_vars_time_3[21] - 0.9475616151022549 x_vars_time_3[24]
   + x_vars_time_4[24] = -1.42565e-08
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
 R94: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= -0
 R95: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= -0
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
End
