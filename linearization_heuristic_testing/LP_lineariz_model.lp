\ Model Linearization Heuristic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 2.939200567528911 u[22] - 2.020604060844469 u[23]
   + 2.9441689374801647e+07 u[29] - 2.22045e-16 u[30]
   - 6.38202087773297 u[32] - 4.387430189149408 u[33]
   + 6.8699835382083461e+07 u[39] - 8.044455473295509 u[42]
   - 5.530299488982911 u[43] + 8.3564516438081369e+07 u[49]
   + 1.11022e-16 u[50] - 6.485364697031439 u[52] - 4.458475677927749 u[53]
   + 6.4228620988678448e+07 u[59] - 1.515563835616438 u[62]
   - 1.041900465911458 u[63] + 1.2321389245678965e+07 u[69] - 1e+06 u[92]
   - 1e+06 u[93] - 1e+06 u[102] - 1e+06 u[103] - 1e+06 u[112]
   - 1e+06 u[113] - 1e+06 u[122] - 1e+06 u[123] - 1e+06 u[132]
   - 1e+06 u[133] - 1e+06 u[142] - 1e+06 u[143] - 1e+06 u[152]
   - 1e+06 u[153] - 1e+06 u[162] - 1e+06 u[163] - 1e+06 u[172]
   - 1e+06 u[173]
Subject To
 H_capacity[t=1]: - u[2] - u[12] - u[22] - u[32] - u[42] - u[52] - u[62]
   - u[72] - u[82] <= 1.9983926158111282e+07
 BH_age_group_0_9[t=1]: u[2] <= 46.52992380000001
 BH_age_group_10_19[t=1]: u[12] <= 45.107730075
 BH_age_group_20_29[t=1]: u[22] <= 164.4689863125
 BH_age_group_30_39[t=1]: u[32] <= 360.34785345
 BH_age_group_40_49[t=1]: u[42] <= 414.5882303625001
 BH_age_group_50_59[t=1]: u[52] <= 731.0444295000001
 BH_age_group_60_69[t=1]: u[62] <= 1085.056575225
 BH_age_group_70_79[t=1]: u[72] <= 1261.7462663625
 BH_age_group_80_plus[t=1]: u[82] <= 3093.84672
 ICU_capacity[t=1]: - u[3] - u[13] - u[23] - u[33] - u[43] - u[53] - u[63]
   - u[73] - u[83] <= -1115.702767930775
 BICU_age_group_0_9[t=1]: u[3] <= 13.2771762
 BICU_age_group_10_19[t=1]: u[13] <= 12.871357425
 BICU_age_group_20_29[t=1]: u[23] <= 21.3716761875
 BICU_age_group_30_39[t=1]: u[33] <= 68.12759654999999
 BICU_age_group_40_49[t=1]: u[43] <= 118.3015258875
 BICU_age_group_50_59[t=1]: u[53] <= 278.6854455000001
 BICU_age_group_60_69[t=1]: u[63] <= 482.944256025
 BICU_age_group_70_79[t=1]: u[73] <= 418.3419711375
 BICU_age_group_80_plus[t=1]: u[83] <= 183.53328
 Mtest_cap[t=1]: u[0] + u[10] + u[20] + u[30] + u[40] + u[50] + u[60]
   + u[70] + u[80] <= 10000
 Atest_cap[t=1]: u[1] + u[11] + u[21] + u[31] + u[41] + u[51] + u[61]
   + u[71] + u[81] <= 10000
 lockdown_age_group_0_9_home[t=1]: u[4] <= 1
 lockdown_age_group_0_9_leisure[t=1]: u[5] <= 1
 lockdown_age_group_0_9_other[t=1]: u[6] <= 1
 lockdown_age_group_0_9_school[t=1]: u[7] <= 1
 lockdown_age_group_0_9_transport[t=1]: u[8] <= 1
 lockdown_age_group_0_9_work[t=1]: u[9] <= 1
 lockdown_age_group_10_19_home[t=1]: u[14] <= 1
 lockdown_age_group_10_19_leisure[t=1]: u[15] <= 1
 lockdown_age_group_10_19_other[t=1]: u[16] <= 1
 lockdown_age_group_10_19_school[t=1]: u[17] <= 1
 lockdown_age_group_10_19_transport[t=1]: u[18] <= 1
 lockdown_age_group_10_19_work[t=1]: u[19] <= 1
 lockdown_age_group_20_29_home[t=1]: u[24] <= 1
 lockdown_age_group_20_29_leisure[t=1]: u[25] <= 1
 lockdown_age_group_20_29_other[t=1]: u[26] <= 1
 lockdown_age_group_20_29_school[t=1]: u[27] <= 1
 lockdown_age_group_20_29_transport[t=1]: u[28] <= 1
 lockdown_age_group_20_29_work[t=1]: u[29] <= 1
 lockdown_age_group_30_39_home[t=1]: u[34] <= 1
 lockdown_age_group_30_39_leisure[t=1]: u[35] <= 1
 lockdown_age_group_30_39_other[t=1]: u[36] <= 1
 lockdown_age_group_30_39_school[t=1]: u[37] <= 1
 lockdown_age_group_30_39_transport[t=1]: u[38] <= 1
 lockdown_age_group_30_39_work[t=1]: u[39] <= 1
 lockdown_age_group_40_49_home[t=1]: u[44] <= 1
 lockdown_age_group_40_49_leisure[t=1]: u[45] <= 1
 lockdown_age_group_40_49_other[t=1]: u[46] <= 1
 lockdown_age_group_40_49_school[t=1]: u[47] <= 1
 lockdown_age_group_40_49_transport[t=1]: u[48] <= 1
 lockdown_age_group_40_49_work[t=1]: u[49] <= 1
 lockdown_age_group_50_59_home[t=1]: u[54] <= 1
 lockdown_age_group_50_59_leisure[t=1]: u[55] <= 1
 lockdown_age_group_50_59_other[t=1]: u[56] <= 1
 lockdown_age_group_50_59_school[t=1]: u[57] <= 1
 lockdown_age_group_50_59_transport[t=1]: u[58] <= 1
 lockdown_age_group_50_59_work[t=1]: u[59] <= 1
 lockdown_age_group_60_69_home[t=1]: u[64] <= 1
 lockdown_age_group_60_69_leisure[t=1]: u[65] <= 1
 lockdown_age_group_60_69_other[t=1]: u[66] <= 1
 lockdown_age_group_60_69_school[t=1]: u[67] <= 1
 lockdown_age_group_60_69_transport[t=1]: u[68] <= 1
 lockdown_age_group_60_69_work[t=1]: u[69] <= 1
 lockdown_age_group_70_79_home[t=1]: u[74] <= 1
 lockdown_age_group_70_79_leisure[t=1]: u[75] <= 1
 lockdown_age_group_70_79_other[t=1]: u[76] <= 1
 lockdown_age_group_70_79_school[t=1]: u[77] <= 1
 lockdown_age_group_70_79_transport[t=1]: u[78] <= 1
 lockdown_age_group_70_79_work[t=1]: u[79] <= 1
 lockdown_age_group_80_plus_home[t=1]: u[84] <= 1
 lockdown_age_group_80_plus_leisure[t=1]: u[85] <= 1
 lockdown_age_group_80_plus_other[t=1]: u[86] <= 1
 lockdown_age_group_80_plus_school[t=1]: u[87] <= 1
 lockdown_age_group_80_plus_transport[t=1]: u[88] <= 1
 lockdown_age_group_80_plus_work[t=1]: u[89] <= 1
 H_capacity[t=2]: - 0.92372234935164 u[2] - 0.92372234935164 u[12]
   - 0.92372234935164 u[22] - 0.9237223493516399 u[32]
   - 0.92372234935164 u[42] - 0.9237223493516399 u[52]
   - 0.92372234935164 u[62] - 0.92372234935164 u[72]
   - 0.92372234935164 u[82] - u[92] - u[102] - u[112] - u[122] - u[132]
   - u[142] - u[152] - u[162] - u[172] <= 1.9963582520341791e+07
 BH_age_group_0_9[t=2]: u[92] <= 81.42736665000002
 BH_age_group_10_19[t=2]: u[102] <= 78.93852763125
 BH_age_group_20_29[t=2]: u[112] <= 296.8395265414941
 BH_age_group_30_39[t=2]: u[122] <= 651.434254038421
 BH_age_group_40_49[t=2]: u[132] <= 749.4519900685063
 BH_age_group_50_59[t=2]: u[142] <= 1309.354873777299
 BH_age_group_60_69[t=2]: u[152] <= 1908.873227184672
 BH_age_group_70_79[t=2]: u[162] <= 2208.055966134375
 BH_age_group_80_plus[t=2]: u[172] <= 5414.231760000001
 ICU_capacity[t=2]: - 0.9475616151022548 u[3] - 0.9475616151022548 u[13]
   - 0.9475616151022548 u[23] - 0.9475616151022549 u[33]
   - 0.9475616151022549 u[43] - 0.9475616151022549 u[53]
   - 0.9475616151022548 u[63] - 0.9475616151022549 u[73]
   - 0.9475616151022549 u[83] - u[93] - u[103] - u[113] - u[123] - u[133]
   - u[143] - u[153] - u[163] - u[173] <= -8287.856783317378
 BICU_age_group_0_9[t=2]: u[93] <= 23.23505835
 BICU_age_group_10_19[t=2]: u[103] <= 22.52487549375
 BICU_age_group_20_29[t=2]: u[113] <= 38.57236785567438
 BICU_age_group_30_39[t=2]: u[123] <= 123.1605783497134
 BICU_age_group_40_49[t=2]: u[133] <= 213.8539097624787
 BICU_age_group_50_59[t=2]: u[143] <= 499.1463330974232
 BICU_age_group_60_69[t=2]: u[153] <= 849.6140953365303
 BICU_age_group_70_79[t=2]: u[163] <= 732.0984494906249
 BICU_age_group_80_plus[t=2]: u[173] <= 321.18324
 Mtest_cap[t=2]: u[90] + u[100] + u[110] + u[120] + u[130] + u[140]
   + u[150] + u[160] + u[170] <= 10000
 Atest_cap[t=2]: u[91] + u[101] + u[111] + u[121] + u[131] + u[141]
   + u[151] + u[161] + u[171] <= 10000
 lockdown_age_group_0_9_home[t=2]: u[94] <= 1
 lockdown_age_group_0_9_leisure[t=2]: u[95] <= 1
 lockdown_age_group_0_9_other[t=2]: u[96] <= 1
 lockdown_age_group_0_9_school[t=2]: u[97] <= 1
 lockdown_age_group_0_9_transport[t=2]: u[98] <= 1
 lockdown_age_group_0_9_work[t=2]: u[99] <= 1
 lockdown_age_group_10_19_home[t=2]: u[104] <= 1
 lockdown_age_group_10_19_leisure[t=2]: u[105] <= 1
 lockdown_age_group_10_19_other[t=2]: u[106] <= 1
 lockdown_age_group_10_19_school[t=2]: u[107] <= 1
 lockdown_age_group_10_19_transport[t=2]: u[108] <= 1
 lockdown_age_group_10_19_work[t=2]: u[109] <= 1
 lockdown_age_group_20_29_home[t=2]: u[114] <= 1
 lockdown_age_group_20_29_leisure[t=2]: u[115] <= 1
 lockdown_age_group_20_29_other[t=2]: u[116] <= 1
 lockdown_age_group_20_29_school[t=2]: u[117] <= 1
 lockdown_age_group_20_29_transport[t=2]: u[118] <= 1
 lockdown_age_group_20_29_work[t=2]: u[119] <= 1
 lockdown_age_group_30_39_home[t=2]: u[124] <= 1
 lockdown_age_group_30_39_leisure[t=2]: u[125] <= 1
 lockdown_age_group_30_39_other[t=2]: u[126] <= 1
 lockdown_age_group_30_39_school[t=2]: u[127] <= 1
 lockdown_age_group_30_39_transport[t=2]: u[128] <= 1
 lockdown_age_group_30_39_work[t=2]: u[129] <= 1
 lockdown_age_group_40_49_home[t=2]: u[134] <= 1
 lockdown_age_group_40_49_leisure[t=2]: u[135] <= 1
 lockdown_age_group_40_49_other[t=2]: u[136] <= 1
 lockdown_age_group_40_49_school[t=2]: u[137] <= 1
 lockdown_age_group_40_49_transport[t=2]: u[138] <= 1
 lockdown_age_group_40_49_work[t=2]: u[139] <= 1
 lockdown_age_group_50_59_home[t=2]: u[144] <= 1
 lockdown_age_group_50_59_leisure[t=2]: u[145] <= 1
 lockdown_age_group_50_59_other[t=2]: u[146] <= 1
 lockdown_age_group_50_59_school[t=2]: u[147] <= 1
 lockdown_age_group_50_59_transport[t=2]: u[148] <= 1
 lockdown_age_group_50_59_work[t=2]: u[149] <= 1
 lockdown_age_group_60_69_home[t=2]: u[154] <= 1
 lockdown_age_group_60_69_leisure[t=2]: u[155] <= 1
 lockdown_age_group_60_69_other[t=2]: u[156] <= 1
 lockdown_age_group_60_69_school[t=2]: u[157] <= 1
 lockdown_age_group_60_69_transport[t=2]: u[158] <= 1
 lockdown_age_group_60_69_work[t=2]: u[159] <= 1
 lockdown_age_group_70_79_home[t=2]: u[164] <= 1
 lockdown_age_group_70_79_leisure[t=2]: u[165] <= 1
 lockdown_age_group_70_79_other[t=2]: u[166] <= 1
 lockdown_age_group_70_79_school[t=2]: u[167] <= 1
 lockdown_age_group_70_79_transport[t=2]: u[168] <= 1
 lockdown_age_group_70_79_work[t=2]: u[169] <= 1
 lockdown_age_group_80_plus_home[t=2]: u[174] <= 1
 lockdown_age_group_80_plus_leisure[t=2]: u[175] <= 1
 lockdown_age_group_80_plus_other[t=2]: u[176] <= 1
 lockdown_age_group_80_plus_school[t=2]: u[177] <= 1
 lockdown_age_group_80_plus_transport[t=2]: u[178] <= 1
 lockdown_age_group_80_plus_work[t=2]: u[179] <= 1
Bounds
End
