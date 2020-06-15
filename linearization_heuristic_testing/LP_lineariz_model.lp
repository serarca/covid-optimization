\ Model Linearization Heuristic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  187.5798632585189 u[20] - 1283.898496616327 u[22] - 897.075338465151 u[23]
   + 2.8962520836583335e+07 u[29] + 407.7305614166262 u[30]
   - 2787.787638862716 u[32] - 1947.860789846432 u[33]
   + 6.7466120530459538e+07 u[39] + 519.266846552055 u[40]
   - 3513.970568175414 u[42] - 2455.253546219042 u[43]
   + 8.2048862022846565e+07 u[49] + 425.0274749398981 u[50]
   - 2832.930177176617 u[52] - 1979.402424908582 u[53]
   + 6.29613290999934e+07 u[59] + 102.2928819220631 u[60]
   - 662.0269986236258 u[62] - 462.5662351256938 u[63]
   + 1.2049379260017533e+07 u[69] + 76.73325493889062 u[110]
   + 110.8747495202981 u[111] - 661.2610023790626 u[112]
   - 458.3944744162034 u[113] + 2.9762208217460107e+07 u[119]
   + 166.8194521963473 u[120] + 241.0436647332799 u[121]
   - 1435.826315984237 u[122] - 995.3329276950968 u[123]
   + 6.9315816668336302e+07 u[129] + 212.4748182935009 u[130]
   + 307.012810500793 u[131] - 1809.840658249942 u[132]
   - 1254.604391202224 u[133] + 8.4289251783081517e+07 u[139]
   + 173.9920623341654 u[140] + 251.407637342949 u[141]
   - 1459.076596449592 u[142] - 1011.450315618475 u[143]
   + 6.4649042505034894e+07 u[149] + 41.91304646337147 u[150]
   + 60.56172818369021 u[151] - 340.9713757478467 u[152]
   - 236.3656619921252 u[153] + 1.2360320185921874e+07 u[159]
   + 21.1772206705166 u[200] + 129.6508943944163 u[201]
   - 229.0334489531793 u[202] - 157.5011595324 u[203]
   + 3.0361973753117688e+07 u[209] + 46.04569901210571 u[210]
   + 281.9003566528909 u[211] - 497.3108229042304 u[212]
   - 341.9894850006834 u[213] + 7.0703088771743879e+07 u[219]
   + 58.65194947719369 u[220] + 359.0781730050696 u[221]
   - 626.8539147528016 u[222] - 431.0733601674636 u[223]
   + 8.5969544103257731e+07 u[229] + 48.04533854262289 u[230]
   + 294.1425227818455 u[231] - 505.3637579857654 u[232]
   - 347.527307614708 u[233] + 6.5914827558816008e+07 u[239]
   + 11.58151225134342 u[240] + 70.90417789889977 u[241]
   - 118.0983755292936 u[242] - 81.21360076343713 u[243]
   + 1.259352588035013e+07 u[249] + 0.2042991136835852 u[290]
   + 86.22935924215311 u[291] - 2.939200567528911 u[292]
   - 2.020604060844469 u[293] + 3.0811797904860873e+07 u[299]
   + 0.4442519910546402 u[300] + 187.5072477799509 u[301]
   - 6.38202087773297 u[302] - 4.387430189149408 u[303]
   + 7.1743542849299565e+07 u[309] + 0.5659098207676196 u[310]
   + 238.8558636099179 u[311] - 8.044455473295509 u[312]
   - 5.530299488982911 u[313] + 8.7229763343389884e+07 u[319]
   + 0.4636884289733088 u[320] + 195.7108643177712 u[321]
   - 6.485364697031439 u[322] - 4.458475677927749 u[323]
   + 6.686416634915185e+07 u[329] + 0.1118307716461764 u[330]
   + 47.20086939554916 u[331] - 1.515563835616438 u[332]
   - 1.041900465911458 u[333] + 1.2768430151171323e+07 u[339]
   - 1e+06 u[362] - 1e+06 u[363] - 1e+06 u[372] - 1e+06 u[373]
   - 1e+06 u[382] - 1e+06 u[383] - 1e+06 u[392] - 1e+06 u[393]
   - 1e+06 u[402] - 1e+06 u[403] - 1e+06 u[412] - 1e+06 u[413]
   - 1e+06 u[422] - 1e+06 u[423] - 1e+06 u[432] - 1e+06 u[433]
   - 1e+06 u[442] - 1e+06 u[443]
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
   - u[142] - u[152] - u[162] - u[172] <= 1.9974322510044944e+07
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
   - u[143] - u[153] - u[163] - u[173] <= -3245.641612655928
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
   <= 1.9947602499065585e+07
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
   <= -8298.399934108076
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
 H_capacity[t=3]: - 0.7881780832918877 u[2] - 0.7881780832918877 u[12]
   - 0.7881780832918877 u[22] - 0.7881780832918875 u[32]
   - 0.7881780832918877 u[42] - 0.7881780832918875 u[52]
   - 0.7881780832918877 u[62] - 0.7881780832918877 u[72]
   - 0.7881780832918877 u[82] - 0.8532629786917132 u[92]
   - 0.8532629786917132 u[102] - 0.8532629786917132 u[112]
   - 0.8532629786917132 u[122] - 0.8532629786917132 u[132]
   - 0.8532629786917132 u[142] - 0.8532629786917132 u[152]
   - 0.8532629786917132 u[162] - 0.8532629786917132 u[172]
   - 0.92372234935164 u[182] - 0.92372234935164 u[192]
   - 0.92372234935164 u[202] - 0.9237223493516399 u[212]
   - 0.92372234935164 u[222] - 0.9237223493516399 u[232]
   - 0.92372234935164 u[242] - 0.92372234935164 u[252]
   - 0.92372234935164 u[262] - u[272] - u[282] - u[292] - u[302] - u[312]
   - u[322] - u[332] - u[342] - u[352] <= 1.9916493795316748e+07
 BH_age_group_0_9[t=3]: u[272] <= 130.8654106875
 BH_age_group_10_19[t=3]: u[282] <= 126.8654908359375
 BH_age_group_20_29[t=3]: u[292] <= 462.5690240039063
 BH_age_group_30_39[t=3]: u[302] <= 1013.478337828125
 BH_age_group_40_49[t=3]: u[312] <= 1166.029397894531
 BH_age_group_50_59[t=3]: u[322] <= 2056.062457968751
 BH_age_group_60_69[t=3]: u[332] <= 3051.721617820312
 BH_age_group_70_79[t=3]: u[342] <= 3548.66137414453
 BH_age_group_80_plus[t=3]: u[352] <= 8701.4439
 ICU_capacity[t=3]: - 0.8507900036959912 u[3] - 0.8507900036959912 u[13]
   - 0.8507900036959912 u[23] - 0.8507900036959912 u[33]
   - 0.8507900036959912 u[43] - 0.8507900036959912 u[53]
   - 0.8507900036959912 u[63] - 0.8507900036959912 u[73]
   - 0.8507900036959912 u[83] - 0.8978730144151937 u[93]
   - 0.8978730144151937 u[103] - 0.8978730144151937 u[113]
   - 0.8978730144151938 u[123] - 0.8978730144151938 u[133]
   - 0.8978730144151938 u[143] - 0.8978730144151937 u[153]
   - 0.8978730144151938 u[163] - 0.8978730144151938 u[173]
   - 0.9475616151022548 u[183] - 0.9475616151022548 u[193]
   - 0.9475616151022548 u[203] - 0.9475616151022549 u[213]
   - 0.9475616151022549 u[223] - 0.9475616151022549 u[233]
   - 0.9475616151022548 u[243] - 0.9475616151022549 u[253]
   - 0.9475616151022549 u[263] - u[273] - u[283] - u[293] - u[303] - u[313]
   - u[323] - u[333] - u[343] - u[353] <= -13638.28780400421
 BICU_age_group_0_9[t=3]: u[273] <= 37.34205806250001
 BICU_age_group_10_19[t=3]: u[283] <= 36.2006927578125
 BICU_age_group_20_29[t=3]: u[293] <= 60.10783927734376
 BICU_age_group_30_39[t=3]: u[303] <= 191.608865296875
 BICU_age_group_40_49[t=3]: u[313] <= 332.7230415585938
 BICU_age_group_50_59[t=3]: u[323] <= 783.8028154687503
 BICU_age_group_60_69[t=3]: u[333] <= 1358.280720070312
 BICU_age_group_70_79[t=3]: u[343] <= 1176.586793824219
 BICU_age_group_80_plus[t=3]: u[353] <= 516.18735
 Mtest_cap[t=3]: u[270] + u[280] + u[290] + u[300] + u[310] + u[320]
   + u[330] + u[340] + u[350] <= 10000
 Atest_cap[t=3]: u[271] + u[281] + u[291] + u[301] + u[311] + u[321]
   + u[331] + u[341] + u[351] <= 10000
 lockdown_age_group_0_9_home[t=3]: u[274] <= 1
 lockdown_age_group_0_9_leisure[t=3]: u[275] <= 1
 lockdown_age_group_0_9_other[t=3]: u[276] <= 1
 lockdown_age_group_0_9_school[t=3]: u[277] <= 1
 lockdown_age_group_0_9_transport[t=3]: u[278] <= 1
 lockdown_age_group_0_9_work[t=3]: u[279] <= 1
 lockdown_age_group_10_19_home[t=3]: u[284] <= 1
 lockdown_age_group_10_19_leisure[t=3]: u[285] <= 1
 lockdown_age_group_10_19_other[t=3]: u[286] <= 1
 lockdown_age_group_10_19_school[t=3]: u[287] <= 1
 lockdown_age_group_10_19_transport[t=3]: u[288] <= 1
 lockdown_age_group_10_19_work[t=3]: u[289] <= 1
 lockdown_age_group_20_29_home[t=3]: u[294] <= 1
 lockdown_age_group_20_29_leisure[t=3]: u[295] <= 1
 lockdown_age_group_20_29_other[t=3]: u[296] <= 1
 lockdown_age_group_20_29_school[t=3]: u[297] <= 1
 lockdown_age_group_20_29_transport[t=3]: u[298] <= 1
 lockdown_age_group_20_29_work[t=3]: u[299] <= 1
 lockdown_age_group_30_39_home[t=3]: u[304] <= 1
 lockdown_age_group_30_39_leisure[t=3]: u[305] <= 1
 lockdown_age_group_30_39_other[t=3]: u[306] <= 1
 lockdown_age_group_30_39_school[t=3]: u[307] <= 1
 lockdown_age_group_30_39_transport[t=3]: u[308] <= 1
 lockdown_age_group_30_39_work[t=3]: u[309] <= 1
 lockdown_age_group_40_49_home[t=3]: u[314] <= 1
 lockdown_age_group_40_49_leisure[t=3]: u[315] <= 1
 lockdown_age_group_40_49_other[t=3]: u[316] <= 1
 lockdown_age_group_40_49_school[t=3]: u[317] <= 1
 lockdown_age_group_40_49_transport[t=3]: u[318] <= 1
 lockdown_age_group_40_49_work[t=3]: u[319] <= 1
 lockdown_age_group_50_59_home[t=3]: u[324] <= 1
 lockdown_age_group_50_59_leisure[t=3]: u[325] <= 1
 lockdown_age_group_50_59_other[t=3]: u[326] <= 1
 lockdown_age_group_50_59_school[t=3]: u[327] <= 1
 lockdown_age_group_50_59_transport[t=3]: u[328] <= 1
 lockdown_age_group_50_59_work[t=3]: u[329] <= 1
 lockdown_age_group_60_69_home[t=3]: u[334] <= 1
 lockdown_age_group_60_69_leisure[t=3]: u[335] <= 1
 lockdown_age_group_60_69_other[t=3]: u[336] <= 1
 lockdown_age_group_60_69_school[t=3]: u[337] <= 1
 lockdown_age_group_60_69_transport[t=3]: u[338] <= 1
 lockdown_age_group_60_69_work[t=3]: u[339] <= 1
 lockdown_age_group_70_79_home[t=3]: u[344] <= 1
 lockdown_age_group_70_79_leisure[t=3]: u[345] <= 1
 lockdown_age_group_70_79_other[t=3]: u[346] <= 1
 lockdown_age_group_70_79_school[t=3]: u[347] <= 1
 lockdown_age_group_70_79_transport[t=3]: u[348] <= 1
 lockdown_age_group_70_79_work[t=3]: u[349] <= 1
 lockdown_age_group_80_plus_home[t=3]: u[354] <= 1
 lockdown_age_group_80_plus_leisure[t=3]: u[355] <= 1
 lockdown_age_group_80_plus_other[t=3]: u[356] <= 1
 lockdown_age_group_80_plus_school[t=3]: u[357] <= 1
 lockdown_age_group_80_plus_transport[t=3]: u[358] <= 1
 lockdown_age_group_80_plus_work[t=3]: u[359] <= 1
 H_capacity[t=4]: - 0.728057710805855 u[2] - 0.728057710805855 u[12]
   - 0.728057710805855 u[22] - 0.728057710805855 u[32]
   - 0.728057710805855 u[42] - 0.728057710805855 u[52]
   - 0.728057710805855 u[62] - 0.728057710805855 u[72]
   - 0.728057710805855 u[82] - 0.7881780832918877 u[92]
   - 0.7881780832918877 u[102] - 0.7881780832918877 u[112]
   - 0.7881780832918875 u[122] - 0.7881780832918877 u[132]
   - 0.7881780832918875 u[142] - 0.7881780832918877 u[152]
   - 0.7881780832918877 u[162] - 0.7881780832918877 u[172]
   - 0.8532629786917132 u[182] - 0.8532629786917132 u[192]
   - 0.8532629786917132 u[202] - 0.8532629786917132 u[212]
   - 0.8532629786917132 u[222] - 0.8532629786917132 u[232]
   - 0.8532629786917132 u[242] - 0.8532629786917132 u[252]
   - 0.8532629786917132 u[262] - 0.92372234935164 u[272]
   - 0.92372234935164 u[282] - 0.92372234935164 u[292]
   - 0.9237223493516399 u[302] - 0.92372234935164 u[312]
   - 0.9237223493516399 u[322] - 0.92372234935164 u[332]
   - 0.92372234935164 u[342] - 0.92372234935164 u[352] - u[362] - u[372]
   - u[382] - u[392] - u[402] - u[412] - u[422] - u[432] - u[442]
   <= 1.9884913431639541e+07
 BH_age_group_0_9[t=4]: u[362] <= 124.322140153125
 BH_age_group_10_19[t=4]: u[372] <= 120.5222162941406
 BH_age_group_20_29[t=4]: u[382] <= 439.4405728037109
 BH_age_group_30_39[t=4]: u[392] <= 962.804420936719
 BH_age_group_40_49[t=4]: u[402] <= 1107.727927999805
 BH_age_group_50_59[t=4]: u[412] <= 1953.259335070313
 BH_age_group_60_69[t=4]: u[422] <= 2899.135536929296
 BH_age_group_70_79[t=4]: u[432] <= 3371.228305437304
 BH_age_group_80_plus[t=4]: u[442] <= 8266.371705
 ICU_capacity[t=4]: - 0.8061759500150267 u[3] - 0.8061759500150267 u[13]
   - 0.8061759500150267 u[23] - 0.8061759500150268 u[33]
   - 0.8061759500150268 u[43] - 0.8061759500150268 u[53]
   - 0.8061759500150267 u[63] - 0.8061759500150268 u[73]
   - 0.8061759500150268 u[83] - 0.8507900036959912 u[93]
   - 0.8507900036959912 u[103] - 0.8507900036959912 u[113]
   - 0.8507900036959912 u[123] - 0.8507900036959912 u[133]
   - 0.8507900036959912 u[143] - 0.8507900036959912 u[153]
   - 0.8507900036959912 u[163] - 0.8507900036959912 u[173]
   - 0.8978730144151937 u[183] - 0.8978730144151937 u[193]
   - 0.8978730144151937 u[203] - 0.8978730144151938 u[213]
   - 0.8978730144151938 u[223] - 0.8978730144151938 u[233]
   - 0.8978730144151937 u[243] - 0.8978730144151938 u[253]
   - 0.8978730144151938 u[263] - 0.9475616151022548 u[273]
   - 0.9475616151022548 u[283] - 0.9475616151022548 u[293]
   - 0.9475616151022549 u[303] - 0.9475616151022549 u[313]
   - 0.9475616151022549 u[323] - 0.9475616151022548 u[333]
   - 0.9475616151022549 u[343] - 0.9475616151022549 u[353] - u[363]
   - u[373] - u[383] - u[393] - u[403] - u[413] - u[423] - u[433] - u[443]
   <= -19809.20438111249
 BICU_age_group_0_9[t=4]: u[363] <= 35.47495515937501
 BICU_age_group_10_19[t=4]: u[373] <= 34.39065811992187
 BICU_age_group_20_29[t=4]: u[383] <= 57.10244731347657
 BICU_age_group_30_39[t=4]: u[393] <= 182.0284220320313
 BICU_age_group_40_49[t=4]: u[403] <= 316.0868894806641
 BICU_age_group_50_59[t=4]: u[413] <= 744.6126746953128
 BICU_age_group_60_69[t=4]: u[423] <= 1290.366684066797
 BICU_age_group_70_79[t=4]: u[433] <= 1117.757454133008
 BICU_age_group_80_plus[t=4]: u[443] <= 490.3779825
 Mtest_cap[t=4]: u[360] + u[370] + u[380] + u[390] + u[400] + u[410]
   + u[420] + u[430] + u[440] <= 10000
 Atest_cap[t=4]: u[361] + u[371] + u[381] + u[391] + u[401] + u[411]
   + u[421] + u[431] + u[441] <= 10000
 lockdown_age_group_0_9_home[t=4]: u[364] <= 1
 lockdown_age_group_0_9_leisure[t=4]: u[365] <= 1
 lockdown_age_group_0_9_other[t=4]: u[366] <= 1
 lockdown_age_group_0_9_school[t=4]: u[367] <= 1
 lockdown_age_group_0_9_transport[t=4]: u[368] <= 1
 lockdown_age_group_0_9_work[t=4]: u[369] <= 1
 lockdown_age_group_10_19_home[t=4]: u[374] <= 1
 lockdown_age_group_10_19_leisure[t=4]: u[375] <= 1
 lockdown_age_group_10_19_other[t=4]: u[376] <= 1
 lockdown_age_group_10_19_school[t=4]: u[377] <= 1
 lockdown_age_group_10_19_transport[t=4]: u[378] <= 1
 lockdown_age_group_10_19_work[t=4]: u[379] <= 1
 lockdown_age_group_20_29_home[t=4]: u[384] <= 1
 lockdown_age_group_20_29_leisure[t=4]: u[385] <= 1
 lockdown_age_group_20_29_other[t=4]: u[386] <= 1
 lockdown_age_group_20_29_school[t=4]: u[387] <= 1
 lockdown_age_group_20_29_transport[t=4]: u[388] <= 1
 lockdown_age_group_20_29_work[t=4]: u[389] <= 1
 lockdown_age_group_30_39_home[t=4]: u[394] <= 1
 lockdown_age_group_30_39_leisure[t=4]: u[395] <= 1
 lockdown_age_group_30_39_other[t=4]: u[396] <= 1
 lockdown_age_group_30_39_school[t=4]: u[397] <= 1
 lockdown_age_group_30_39_transport[t=4]: u[398] <= 1
 lockdown_age_group_30_39_work[t=4]: u[399] <= 1
 lockdown_age_group_40_49_home[t=4]: u[404] <= 1
 lockdown_age_group_40_49_leisure[t=4]: u[405] <= 1
 lockdown_age_group_40_49_other[t=4]: u[406] <= 1
 lockdown_age_group_40_49_school[t=4]: u[407] <= 1
 lockdown_age_group_40_49_transport[t=4]: u[408] <= 1
 lockdown_age_group_40_49_work[t=4]: u[409] <= 1
 lockdown_age_group_50_59_home[t=4]: u[414] <= 1
 lockdown_age_group_50_59_leisure[t=4]: u[415] <= 1
 lockdown_age_group_50_59_other[t=4]: u[416] <= 1
 lockdown_age_group_50_59_school[t=4]: u[417] <= 1
 lockdown_age_group_50_59_transport[t=4]: u[418] <= 1
 lockdown_age_group_50_59_work[t=4]: u[419] <= 1
 lockdown_age_group_60_69_home[t=4]: u[424] <= 1
 lockdown_age_group_60_69_leisure[t=4]: u[425] <= 1
 lockdown_age_group_60_69_other[t=4]: u[426] <= 1
 lockdown_age_group_60_69_school[t=4]: u[427] <= 1
 lockdown_age_group_60_69_transport[t=4]: u[428] <= 1
 lockdown_age_group_60_69_work[t=4]: u[429] <= 1
 lockdown_age_group_70_79_home[t=4]: u[434] <= 1
 lockdown_age_group_70_79_leisure[t=4]: u[435] <= 1
 lockdown_age_group_70_79_other[t=4]: u[436] <= 1
 lockdown_age_group_70_79_school[t=4]: u[437] <= 1
 lockdown_age_group_70_79_transport[t=4]: u[438] <= 1
 lockdown_age_group_70_79_work[t=4]: u[439] <= 1
 lockdown_age_group_80_plus_home[t=4]: u[444] <= 1
 lockdown_age_group_80_plus_leisure[t=4]: u[445] <= 1
 lockdown_age_group_80_plus_other[t=4]: u[446] <= 1
 lockdown_age_group_80_plus_school[t=4]: u[447] <= 1
 lockdown_age_group_80_plus_transport[t=4]: u[448] <= 1
 lockdown_age_group_80_plus_work[t=4]: u[449] <= 1
Bounds
End
