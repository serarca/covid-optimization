\ Model Linearization Heuristic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  37.63850960016097 u[20] - 229.0334489531793 u[22] - 157.5011595324 u[23]
   + 2.8962520836583335e+07 u[29] + 81.81246314807575 u[30]
   - 497.3108229042304 u[32] - 341.9894850006834 u[33]
   + 6.7466120530459538e+07 u[39] + 104.1925814929242 u[40]
   - 626.8539147528016 u[42] - 431.0733601674636 u[43]
   + 8.2048862022846565e+07 u[49] + 85.28314509863 u[50]
   - 505.3637579857654 u[52] - 347.527307614708 u[53]
   + 6.29613290999934e+07 u[59] + 20.52539942917807 u[60]
   - 118.0983755292936 u[62] - 81.21360076343713 u[63]
   + 1.2049379260017533e+07 u[69] + 0.3631269089092719 u[110]
   + 37.28102931468519 u[111] - 2.939200567528911 u[112]
   - 2.020604060844469 u[113] + 2.9762208217460107e+07 u[119]
   + 0.7894443168120537 u[120] + 81.04961652603741 u[121]
   - 6.38202087773297 u[122] - 4.387430189149408 u[123]
   + 6.9315816668336302e+07 u[129] + 1.005500471072466 u[130]
   + 103.2313816967733 u[131] - 8.044455473295509 u[132]
   - 5.530299488982911 u[133] + 8.4289251783081517e+07 u[139]
   + 0.8233874585467763 u[140] + 84.53444574413568 u[141]
   - 6.485364697031439 u[142] - 4.458475677927749 u[143]
   + 6.4649042505034894e+07 u[149] + 0.1983462713439643 u[150]
   + 20.36355052464699 u[151] - 1.515563835616438 u[152]
   - 1.041900465911458 u[153] + 1.2360320185921874e+07 u[159]
   - 1e+06 u[182] - 1e+06 u[183] - 1e+06 u[192] - 1e+06 u[193]
   - 1e+06 u[202] - 1e+06 u[203] - 1e+06 u[212] - 1e+06 u[213]
   - 1e+06 u[222] - 1e+06 u[223] - 1e+06 u[232] - 1e+06 u[233]
   - 1e+06 u[242] - 1e+06 u[243] - 1e+06 u[252] - 1e+06 u[253]
   - 1e+06 u[262] - 1e+06 u[263]
Subject To
 H_capacity[t=0]: - u[2] - u[12] - u[22] - u[32] - u[42] - u[52] - u[62]
   - u[72] - u[82] <= 1.9990396351046551e+07
 BH_age_group_0_9[t=0]: u[2] <= 62.03989840000001
 BH_age_group_10_19[t=0]: u[12] <= 60.1436401
 BH_age_group_20_29[t=0]: u[22] <= 219.29198175
 BH_age_group_30_39[t=0]: u[32] <= 480.4638046000001
 BH_age_group_40_49[t=0]: u[42] <= 552.7843071500001
 BH_age_group_50_59[t=0]: u[52] <= 974.7259060000001
 BH_age_group_60_69[t=0]: u[62] <= 1446.7421003
 BH_age_group_70_79[t=0]: u[72] <= 1682.32835515
 BH_age_group_80_plus[t=0]: u[82] <= 4125.12896
 ICU_capacity[t=0]: - u[3] - u[13] - u[23] - u[33] - u[43] - u[53] - u[63]
   - u[73] - u[83] <= 370.0609534499999
 BICU_age_group_0_9[t=0]: u[3] <= 17.7029016
 BICU_age_group_10_19[t=0]: u[13] <= 17.1618099
 BICU_age_group_20_29[t=0]: u[23] <= 28.49556825000001
 BICU_age_group_30_39[t=0]: u[33] <= 90.8367954
 BICU_age_group_40_49[t=0]: u[43] <= 157.73536785
 BICU_age_group_50_59[t=0]: u[53] <= 371.5805940000001
 BICU_age_group_60_69[t=0]: u[63] <= 643.9256746999999
 BICU_age_group_70_79[t=0]: u[73] <= 557.78929485
 BICU_age_group_80_plus[t=0]: u[83] <= 244.71104
 Mtest_cap[t=0]: u[0] + u[10] + u[20] + u[30] + u[40] + u[50] + u[60]
   + u[70] + u[80] <= 10000
 Atest_cap[t=0]: u[1] + u[11] + u[21] + u[31] + u[41] + u[51] + u[61]
   + u[71] + u[81] <= 10000
 lockdown_age_group_0_9_home[t=0]: u[4] <= 1
 lockdown_age_group_0_9_leisure[t=0]: u[5] <= 1
 lockdown_age_group_0_9_other[t=0]: u[6] <= 1
 lockdown_age_group_0_9_school[t=0]: u[7] <= 1
 lockdown_age_group_0_9_transport[t=0]: u[8] <= 1
 lockdown_age_group_0_9_work[t=0]: u[9] <= 1
 lockdown_age_group_10_19_home[t=0]: u[14] <= 1
 lockdown_age_group_10_19_leisure[t=0]: u[15] <= 1
 lockdown_age_group_10_19_other[t=0]: u[16] <= 1
 lockdown_age_group_10_19_school[t=0]: u[17] <= 1
 lockdown_age_group_10_19_transport[t=0]: u[18] <= 1
 lockdown_age_group_10_19_work[t=0]: u[19] <= 1
 lockdown_age_group_20_29_home[t=0]: u[24] <= 1
 lockdown_age_group_20_29_leisure[t=0]: u[25] <= 1
 lockdown_age_group_20_29_other[t=0]: u[26] <= 1
 lockdown_age_group_20_29_school[t=0]: u[27] <= 1
 lockdown_age_group_20_29_transport[t=0]: u[28] <= 1
 lockdown_age_group_20_29_work[t=0]: u[29] <= 1
 lockdown_age_group_30_39_home[t=0]: u[34] <= 1
 lockdown_age_group_30_39_leisure[t=0]: u[35] <= 1
 lockdown_age_group_30_39_other[t=0]: u[36] <= 1
 lockdown_age_group_30_39_school[t=0]: u[37] <= 1
 lockdown_age_group_30_39_transport[t=0]: u[38] <= 1
 lockdown_age_group_30_39_work[t=0]: u[39] <= 1
 lockdown_age_group_40_49_home[t=0]: u[44] <= 1
 lockdown_age_group_40_49_leisure[t=0]: u[45] <= 1
 lockdown_age_group_40_49_other[t=0]: u[46] <= 1
 lockdown_age_group_40_49_school[t=0]: u[47] <= 1
 lockdown_age_group_40_49_transport[t=0]: u[48] <= 1
 lockdown_age_group_40_49_work[t=0]: u[49] <= 1
 lockdown_age_group_50_59_home[t=0]: u[54] <= 1
 lockdown_age_group_50_59_leisure[t=0]: u[55] <= 1
 lockdown_age_group_50_59_other[t=0]: u[56] <= 1
 lockdown_age_group_50_59_school[t=0]: u[57] <= 1
 lockdown_age_group_50_59_transport[t=0]: u[58] <= 1
 lockdown_age_group_50_59_work[t=0]: u[59] <= 1
 lockdown_age_group_60_69_home[t=0]: u[64] <= 1
 lockdown_age_group_60_69_leisure[t=0]: u[65] <= 1
 lockdown_age_group_60_69_other[t=0]: u[66] <= 1
 lockdown_age_group_60_69_school[t=0]: u[67] <= 1
 lockdown_age_group_60_69_transport[t=0]: u[68] <= 1
 lockdown_age_group_60_69_work[t=0]: u[69] <= 1
 lockdown_age_group_70_79_home[t=0]: u[74] <= 1
 lockdown_age_group_70_79_leisure[t=0]: u[75] <= 1
 lockdown_age_group_70_79_other[t=0]: u[76] <= 1
 lockdown_age_group_70_79_school[t=0]: u[77] <= 1
 lockdown_age_group_70_79_transport[t=0]: u[78] <= 1
 lockdown_age_group_70_79_work[t=0]: u[79] <= 1
 lockdown_age_group_80_plus_home[t=0]: u[84] <= 1
 lockdown_age_group_80_plus_leisure[t=0]: u[85] <= 1
 lockdown_age_group_80_plus_other[t=0]: u[86] <= 1
 lockdown_age_group_80_plus_school[t=0]: u[87] <= 1
 lockdown_age_group_80_plus_transport[t=0]: u[88] <= 1
 lockdown_age_group_80_plus_work[t=0]: u[89] <= 1
 H_capacity[t=1]: - 0.92372234935164 u[2] - 0.92372234935164 u[12]
   - 0.92372234935164 u[22] - 0.9237223493516399 u[32]
   - 0.92372234935164 u[42] - 0.9237223493516399 u[52]
   - 0.92372234935164 u[62] - 0.92372234935164 u[72]
   - 0.92372234935164 u[82] - u[92] - u[102] - u[112] - u[122] - u[132]
   - u[142] - u[152] - u[162] - u[172] <= 1.9974322509157833e+07
 BH_age_group_0_9[t=1]: u[92] <= 108.5698222
 BH_age_group_10_19[t=1]: u[102] <= 105.251370175
 BH_age_group_20_29[t=1]: u[112] <= 383.7609680625001
 BH_age_group_30_39[t=1]: u[122] <= 840.81165805
 BH_age_group_40_49[t=1]: u[132] <= 967.3725375125002
 BH_age_group_50_59[t=1]: u[142] <= 1705.7703355
 BH_age_group_60_69[t=1]: u[152] <= 2531.798675525
 BH_age_group_70_79[t=1]: u[162] <= 2944.074621512499
 BH_age_group_80_plus[t=1]: u[172] <= 7218.97568
 ICU_capacity[t=1]: - 0.9475616151022548 u[3] - 0.9475616151022548 u[13]
   - 0.9475616151022548 u[23] - 0.9475616151022549 u[33]
   - 0.9475616151022549 u[43] - 0.9475616151022549 u[53]
   - 0.9475616151022548 u[63] - 0.9475616151022549 u[73]
   - 0.9475616151022549 u[83] - u[93] - u[103] - u[113] - u[123] - u[133]
   - u[143] - u[153] - u[163] - u[173] <= -3245.641814480775
 BICU_age_group_0_9[t=1]: u[93] <= 30.98007780000001
 BICU_age_group_10_19[t=1]: u[103] <= 30.03316732499999
 BICU_age_group_20_29[t=1]: u[113] <= 49.86724443750001
 BICU_age_group_30_39[t=1]: u[123] <= 158.96439195
 BICU_age_group_40_49[t=1]: u[133] <= 276.0368937375001
 BICU_age_group_50_59[t=1]: u[143] <= 650.2660395000003
 BICU_age_group_60_69[t=1]: u[153] <= 1126.869930725
 BICU_age_group_70_79[t=1]: u[163] <= 976.1312659874999
 BICU_age_group_80_plus[t=1]: u[173] <= 428.2443200000001
 Mtest_cap[t=1]: u[90] + u[100] + u[110] + u[120] + u[130] + u[140]
   + u[150] + u[160] + u[170] <= 10000
 Atest_cap[t=1]: u[91] + u[101] + u[111] + u[121] + u[131] + u[141]
   + u[151] + u[161] + u[171] <= 10000
 lockdown_age_group_0_9_home[t=1]: u[94] <= 1
 lockdown_age_group_0_9_leisure[t=1]: u[95] <= 1
 lockdown_age_group_0_9_other[t=1]: u[96] <= 1
 lockdown_age_group_0_9_school[t=1]: u[97] <= 1
 lockdown_age_group_0_9_transport[t=1]: u[98] <= 1
 lockdown_age_group_0_9_work[t=1]: u[99] <= 1
 lockdown_age_group_10_19_home[t=1]: u[104] <= 1
 lockdown_age_group_10_19_leisure[t=1]: u[105] <= 1
 lockdown_age_group_10_19_other[t=1]: u[106] <= 1
 lockdown_age_group_10_19_school[t=1]: u[107] <= 1
 lockdown_age_group_10_19_transport[t=1]: u[108] <= 1
 lockdown_age_group_10_19_work[t=1]: u[109] <= 1
 lockdown_age_group_20_29_home[t=1]: u[114] <= 1
 lockdown_age_group_20_29_leisure[t=1]: u[115] <= 1
 lockdown_age_group_20_29_other[t=1]: u[116] <= 1
 lockdown_age_group_20_29_school[t=1]: u[117] <= 1
 lockdown_age_group_20_29_transport[t=1]: u[118] <= 1
 lockdown_age_group_20_29_work[t=1]: u[119] <= 1
 lockdown_age_group_30_39_home[t=1]: u[124] <= 1
 lockdown_age_group_30_39_leisure[t=1]: u[125] <= 1
 lockdown_age_group_30_39_other[t=1]: u[126] <= 1
 lockdown_age_group_30_39_school[t=1]: u[127] <= 1
 lockdown_age_group_30_39_transport[t=1]: u[128] <= 1
 lockdown_age_group_30_39_work[t=1]: u[129] <= 1
 lockdown_age_group_40_49_home[t=1]: u[134] <= 1
 lockdown_age_group_40_49_leisure[t=1]: u[135] <= 1
 lockdown_age_group_40_49_other[t=1]: u[136] <= 1
 lockdown_age_group_40_49_school[t=1]: u[137] <= 1
 lockdown_age_group_40_49_transport[t=1]: u[138] <= 1
 lockdown_age_group_40_49_work[t=1]: u[139] <= 1
 lockdown_age_group_50_59_home[t=1]: u[144] <= 1
 lockdown_age_group_50_59_leisure[t=1]: u[145] <= 1
 lockdown_age_group_50_59_other[t=1]: u[146] <= 1
 lockdown_age_group_50_59_school[t=1]: u[147] <= 1
 lockdown_age_group_50_59_transport[t=1]: u[148] <= 1
 lockdown_age_group_50_59_work[t=1]: u[149] <= 1
 lockdown_age_group_60_69_home[t=1]: u[154] <= 1
 lockdown_age_group_60_69_leisure[t=1]: u[155] <= 1
 lockdown_age_group_60_69_other[t=1]: u[156] <= 1
 lockdown_age_group_60_69_school[t=1]: u[157] <= 1
 lockdown_age_group_60_69_transport[t=1]: u[158] <= 1
 lockdown_age_group_60_69_work[t=1]: u[159] <= 1
 lockdown_age_group_70_79_home[t=1]: u[164] <= 1
 lockdown_age_group_70_79_leisure[t=1]: u[165] <= 1
 lockdown_age_group_70_79_other[t=1]: u[166] <= 1
 lockdown_age_group_70_79_school[t=1]: u[167] <= 1
 lockdown_age_group_70_79_transport[t=1]: u[168] <= 1
 lockdown_age_group_70_79_work[t=1]: u[169] <= 1
 lockdown_age_group_80_plus_home[t=1]: u[174] <= 1
 lockdown_age_group_80_plus_leisure[t=1]: u[175] <= 1
 lockdown_age_group_80_plus_other[t=1]: u[176] <= 1
 lockdown_age_group_80_plus_school[t=1]: u[177] <= 1
 lockdown_age_group_80_plus_transport[t=1]: u[178] <= 1
 lockdown_age_group_80_plus_work[t=1]: u[179] <= 1
 H_capacity[t=2]: - 0.8532629786917132 u[2] - 0.8532629786917132 u[12]
   - 0.8532629786917132 u[22] - 0.8532629786917132 u[32]
   - 0.8532629786917132 u[42] - 0.8532629786917132 u[52]
   - 0.8532629786917132 u[62] - 0.8532629786917132 u[72]
   - 0.8532629786917132 u[82] - 0.92372234935164 u[92]
   - 0.92372234935164 u[102] - 0.92372234935164 u[112]
   - 0.9237223493516399 u[122] - 0.92372234935164 u[132]
   - 0.9237223493516399 u[142] - 0.92372234935164 u[152]
   - 0.92372234935164 u[162] - 0.92372234935164 u[172] - u[182] - u[192]
   - u[202] - u[212] - u[222] - u[232] - u[242] - u[252] - u[262]
   <= 1.9947602496693697e+07
 BH_age_group_0_9[t=2]: u[182] <= 127.95729045
 BH_age_group_10_19[t=2]: u[192] <= 124.04625770625
 BH_age_group_20_29[t=2]: u[202] <= 452.2897123593751
 BH_age_group_30_39[t=2]: u[212] <= 990.9565969875
 BH_age_group_40_49[t=2]: u[222] <= 1140.117633496875
 BH_age_group_50_59[t=2]: u[232] <= 2010.372181125
 BH_age_group_60_69[t=2]: u[242] <= 2983.90558186875
 BH_age_group_70_79[t=2]: u[252] <= 3469.802232496874
 BH_age_group_80_plus[t=2]: u[262] <= 8508.07848
 ICU_capacity[t=2]: - 0.8978730144151937 u[3] - 0.8978730144151937 u[13]
   - 0.8978730144151937 u[23] - 0.8978730144151938 u[33]
   - 0.8978730144151938 u[43] - 0.8978730144151938 u[53]
   - 0.8978730144151937 u[63] - 0.8978730144151938 u[73]
   - 0.8978730144151938 u[83] - 0.9475616151022548 u[93]
   - 0.9475616151022548 u[103] - 0.9475616151022548 u[113]
   - 0.9475616151022549 u[123] - 0.9475616151022549 u[133]
   - 0.9475616151022549 u[143] - 0.9475616151022548 u[153]
   - 0.9475616151022549 u[163] - 0.9475616151022549 u[173] - u[183]
   - u[193] - u[203] - u[213] - u[223] - u[233] - u[243] - u[253] - u[263]
   <= -8351.232777382431
 BICU_age_group_0_9[t=2]: u[183] <= 36.51223455
 BICU_age_group_10_19[t=2]: u[193] <= 35.39623291874999
 BICU_age_group_20_29[t=2]: u[203] <= 58.77210951562501
 BICU_age_group_30_39[t=2]: u[213] <= 187.3508905125
 BICU_age_group_40_49[t=2]: u[223] <= 325.3291961906251
 BICU_age_group_50_59[t=2]: u[233] <= 766.3849751250002
 BICU_age_group_60_69[t=2]: u[243] <= 1328.09670406875
 BICU_age_group_70_79[t=2]: u[253] <= 1150.440420628125
 BICU_age_group_80_plus[t=2]: u[263] <= 504.7165200000001
 Mtest_cap[t=2]: u[180] + u[190] + u[200] + u[210] + u[220] + u[230]
   + u[240] + u[250] + u[260] <= 10000
 Atest_cap[t=2]: u[181] + u[191] + u[201] + u[211] + u[221] + u[231]
   + u[241] + u[251] + u[261] <= 10000
 lockdown_age_group_0_9_home[t=2]: u[184] <= 1
 lockdown_age_group_0_9_leisure[t=2]: u[185] <= 1
 lockdown_age_group_0_9_other[t=2]: u[186] <= 1
 lockdown_age_group_0_9_school[t=2]: u[187] <= 1
 lockdown_age_group_0_9_transport[t=2]: u[188] <= 1
 lockdown_age_group_0_9_work[t=2]: u[189] <= 1
 lockdown_age_group_10_19_home[t=2]: u[194] <= 1
 lockdown_age_group_10_19_leisure[t=2]: u[195] <= 1
 lockdown_age_group_10_19_other[t=2]: u[196] <= 1
 lockdown_age_group_10_19_school[t=2]: u[197] <= 1
 lockdown_age_group_10_19_transport[t=2]: u[198] <= 1
 lockdown_age_group_10_19_work[t=2]: u[199] <= 1
 lockdown_age_group_20_29_home[t=2]: u[204] <= 1
 lockdown_age_group_20_29_leisure[t=2]: u[205] <= 1
 lockdown_age_group_20_29_other[t=2]: u[206] <= 1
 lockdown_age_group_20_29_school[t=2]: u[207] <= 1
 lockdown_age_group_20_29_transport[t=2]: u[208] <= 1
 lockdown_age_group_20_29_work[t=2]: u[209] <= 1
 lockdown_age_group_30_39_home[t=2]: u[214] <= 1
 lockdown_age_group_30_39_leisure[t=2]: u[215] <= 1
 lockdown_age_group_30_39_other[t=2]: u[216] <= 1
 lockdown_age_group_30_39_school[t=2]: u[217] <= 1
 lockdown_age_group_30_39_transport[t=2]: u[218] <= 1
 lockdown_age_group_30_39_work[t=2]: u[219] <= 1
 lockdown_age_group_40_49_home[t=2]: u[224] <= 1
 lockdown_age_group_40_49_leisure[t=2]: u[225] <= 1
 lockdown_age_group_40_49_other[t=2]: u[226] <= 1
 lockdown_age_group_40_49_school[t=2]: u[227] <= 1
 lockdown_age_group_40_49_transport[t=2]: u[228] <= 1
 lockdown_age_group_40_49_work[t=2]: u[229] <= 1
 lockdown_age_group_50_59_home[t=2]: u[234] <= 1
 lockdown_age_group_50_59_leisure[t=2]: u[235] <= 1
 lockdown_age_group_50_59_other[t=2]: u[236] <= 1
 lockdown_age_group_50_59_school[t=2]: u[237] <= 1
 lockdown_age_group_50_59_transport[t=2]: u[238] <= 1
 lockdown_age_group_50_59_work[t=2]: u[239] <= 1
 lockdown_age_group_60_69_home[t=2]: u[244] <= 1
 lockdown_age_group_60_69_leisure[t=2]: u[245] <= 1
 lockdown_age_group_60_69_other[t=2]: u[246] <= 1
 lockdown_age_group_60_69_school[t=2]: u[247] <= 1
 lockdown_age_group_60_69_transport[t=2]: u[248] <= 1
 lockdown_age_group_60_69_work[t=2]: u[249] <= 1
 lockdown_age_group_70_79_home[t=2]: u[254] <= 1
 lockdown_age_group_70_79_leisure[t=2]: u[255] <= 1
 lockdown_age_group_70_79_other[t=2]: u[256] <= 1
 lockdown_age_group_70_79_school[t=2]: u[257] <= 1
 lockdown_age_group_70_79_transport[t=2]: u[258] <= 1
 lockdown_age_group_70_79_work[t=2]: u[259] <= 1
 lockdown_age_group_80_plus_home[t=2]: u[264] <= 1
 lockdown_age_group_80_plus_leisure[t=2]: u[265] <= 1
 lockdown_age_group_80_plus_other[t=2]: u[266] <= 1
 lockdown_age_group_80_plus_school[t=2]: u[267] <= 1
 lockdown_age_group_80_plus_transport[t=2]: u[268] <= 1
 lockdown_age_group_80_plus_work[t=2]: u[269] <= 1
Bounds
End
