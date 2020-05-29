\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 100000 ICU-Bounces[0,age_group_0_9]
   - 100000 ICU-Bounces[0,age_group_10_19]
   - 100000 ICU-Bounces[0,age_group_20_29]
   - 100000 ICU-Bounces[0,age_group_30_39]
   - 100000 ICU-Bounces[0,age_group_40_49]
   - 100000 ICU-Bounces[0,age_group_50_59]
   - 100000 ICU-Bounces[0,age_group_60_69]
   - 100000 ICU-Bounces[0,age_group_70_79]
   - 100000 ICU-Bounces[0,age_group_80_plus]
   - 100000 H-Bounces[0,age_group_0_9]
   - 100000 H-Bounces[0,age_group_10_19]
   - 100000 H-Bounces[0,age_group_20_29]
   - 100000 H-Bounces[0,age_group_30_39]
   - 100000 H-Bounces[0,age_group_40_49]
   - 100000 H-Bounces[0,age_group_50_59]
   - 100000 H-Bounces[0,age_group_60_69]
   - 100000 H-Bounces[0,age_group_70_79]
   - 100000 H-Bounces[0,age_group_80_plus]
   + 5.0663783764688736e+08 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.9996019554175865e+07
 ICUCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.9998428863902789e+07
 R2: H-Bounces[0,age_group_0_9] <= 2.958534760231826
 R3: H-Bounces[0,age_group_10_19] <= 2.910842758699157
 R4: H-Bounces[0,age_group_20_29] <= 9.263646756921602
 R5: H-Bounces[0,age_group_30_39] <= 33.37429400563177
 R6: H-Bounces[0,age_group_40_49] <= 38.42873845621118
 R7: H-Bounces[0,age_group_50_59] <= 54.12185288445627
 R8: H-Bounces[0,age_group_60_69] <= 48.0166264998989
 R9: H-Bounces[0,age_group_70_79] <= 39.24366576391524
 R10: H-Bounces[0,age_group_80_plus] <= 64.00555650116472
 R11: ICU-Bounces[0,age_group_0_9] <= 0.8442091475211636
 R12: ICU-Bounces[0,age_group_10_19] <= 0.830600375875595
 R13: ICU-Bounces[0,age_group_20_29] <= 1.203750708526536
 R14: ICU-Bounces[0,age_group_30_39] <= 6.309765454096851
 R15: ICU-Bounces[0,age_group_40_49] <= 10.96552691166951
 R16: ICU-Bounces[0,age_group_50_59] <= 20.63208756368775
 R17: ICU-Bounces[0,age_group_60_69] <= 21.37156208377003
 R18: ICU-Bounces[0,age_group_70_79] <= 13.01154830254979
 R19: ICU-Bounces[0,age_group_80_plus] <= 3.796939792441975
 R20: m-tests[0,age_group_0_9] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_20_29] + m-tests[0,age_group_30_39]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_50_59]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_80_plus] <= 100000
 R21: a-tests[0,age_group_0_9] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_20_29] + a-tests[0,age_group_30_39]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_50_59]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_80_plus] <= 100000
Bounds
 a-tests[0,age_group_0_9] <= 100000
 a-tests[0,age_group_10_19] <= 100000
 a-tests[0,age_group_20_29] <= 100000
 a-tests[0,age_group_30_39] <= 100000
 a-tests[0,age_group_40_49] <= 100000
 a-tests[0,age_group_50_59] <= 100000
 a-tests[0,age_group_60_69] <= 100000
 a-tests[0,age_group_70_79] <= 100000
 a-tests[0,age_group_80_plus] <= 100000
 m-tests[0,age_group_0_9] <= 100000
 m-tests[0,age_group_10_19] <= 100000
 m-tests[0,age_group_20_29] <= 100000
 m-tests[0,age_group_30_39] <= 100000
 m-tests[0,age_group_40_49] <= 100000
 m-tests[0,age_group_50_59] <= 100000
 m-tests[0,age_group_60_69] <= 100000
 m-tests[0,age_group_70_79] <= 100000
 m-tests[0,age_group_80_plus] <= 100000
 Constant = 1
End
