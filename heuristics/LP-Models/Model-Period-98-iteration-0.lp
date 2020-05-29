\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 99968.53696906136 ICU-Bounces[0,age_group_0_9]
   - 99968.53696906136 ICU-Bounces[0,age_group_10_19]
   - 99942.31777661249 ICU-Bounces[0,age_group_20_29]
   - 99900.36706869428 ICU-Bounces[0,age_group_30_39]
   - 99826.95332983744 ICU-Bounces[0,age_group_40_49]
   - 99659.15049816466 ICU-Bounces[0,age_group_50_59]
   - 99339.2763502884 ICU-Bounces[0,age_group_60_69]
   - 98898.79391714736 ICU-Bounces[0,age_group_70_79]
   - 98342.94703723125 ICU-Bounces[0,age_group_80_plus]
   - 100000 ICU-Bounces[1,age_group_0_9]
   - 100000 ICU-Bounces[1,age_group_10_19]
   - 100000 ICU-Bounces[1,age_group_20_29]
   - 100000 ICU-Bounces[1,age_group_30_39]
   - 100000 ICU-Bounces[1,age_group_40_49]
   - 100000 ICU-Bounces[1,age_group_50_59]
   - 100000 ICU-Bounces[1,age_group_60_69]
   - 100000 ICU-Bounces[1,age_group_70_79]
   - 100000 ICU-Bounces[1,age_group_80_plus]
   - 99954.23340961098 H-Bounces[0,age_group_0_9]
   - 99954.23340961098 H-Bounces[0,age_group_10_19]
   - 99916.09458428681 H-Bounces[0,age_group_20_29]
   - 99855.07246376811 H-Bounces[0,age_group_30_39]
   - 99748.28375286041 H-Bounces[0,age_group_40_49]
   - 99504.19527078565 H-Bounces[0,age_group_50_59]
   - 99038.90160183066 H-Bounces[0,age_group_60_69]
   - 98398.16933638444 H-Bounces[0,age_group_70_79]
   - 97589.62623951182 H-Bounces[0,age_group_80_plus]
   - 100000 H-Bounces[1,age_group_0_9]
   - 100000 H-Bounces[1,age_group_10_19]
   - 100000 H-Bounces[1,age_group_20_29]
   - 100000 H-Bounces[1,age_group_30_39]
   - 100000 H-Bounces[1,age_group_40_49]
   - 100000 H-Bounces[1,age_group_50_59]
   - 100000 H-Bounces[1,age_group_60_69]
   - 100000 H-Bounces[1,age_group_70_79]
   - 100000 H-Bounces[1,age_group_80_plus]
   + 0.6570763969146219 a-tests[0,age_group_20_29]
   + 0.6583386882064142 a-tests[0,age_group_30_39]
   + 0.2234396756277768 a-tests[0,age_group_40_49]
   + 1.013124238647977e+09 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.9995995303190757e+07
 HCAP[1]: - 0.92372234935164 H-Bounces[0,age_group_0_9]
   - 0.92372234935164 H-Bounces[0,age_group_10_19]
   - 0.92372234935164 H-Bounces[0,age_group_20_29]
   - 0.9237223493516399 H-Bounces[0,age_group_30_39]
   - 0.92372234935164 H-Bounces[0,age_group_40_49]
   - 0.9237223493516399 H-Bounces[0,age_group_50_59]
   - 0.92372234935164 H-Bounces[0,age_group_60_69]
   - 0.92372234935164 H-Bounces[0,age_group_70_79]
   - 0.92372234935164 H-Bounces[0,age_group_80_plus]
   - H-Bounces[1,age_group_0_9] - H-Bounces[1,age_group_10_19]
   - H-Bounces[1,age_group_20_29] - H-Bounces[1,age_group_30_39]
   - H-Bounces[1,age_group_40_49] - H-Bounces[1,age_group_50_59]
   - H-Bounces[1,age_group_60_69] - H-Bounces[1,age_group_70_79]
   - H-Bounces[1,age_group_80_plus] + 5.57126e-06 m-tests[0,age_group_0_9]
   + 5.66408e-06 m-tests[0,age_group_10_19]
   + 1.71563e-05 m-tests[0,age_group_20_29]
   + 6.43706e-05 m-tests[0,age_group_30_39]
   + 6.93059e-05 m-tests[0,age_group_40_49]
   + 1.062725048463596e-04 m-tests[0,age_group_50_59]
   + 1.2327228319996286e-04 m-tests[0,age_group_60_69]
   + 0.000150013 m-tests[0,age_group_70_79]
   + 2.4535611000052834e-04 m-tests[0,age_group_80_plus]
   <= 1.9996019554175865e+07
 ICUCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.9998424434969909e+07
 ICUCAP[1]: - 0.9475616151022548 ICU-Bounces[0,age_group_0_9]
   - 0.9475616151022548 ICU-Bounces[0,age_group_10_19]
   - 0.9475616151022548 ICU-Bounces[0,age_group_20_29]
   - 0.9475616151022549 ICU-Bounces[0,age_group_30_39]
   - 0.9475616151022549 ICU-Bounces[0,age_group_40_49]
   - 0.9475616151022549 ICU-Bounces[0,age_group_50_59]
   - 0.9475616151022548 ICU-Bounces[0,age_group_60_69]
   - 0.9475616151022549 ICU-Bounces[0,age_group_70_79]
   - 0.9475616151022549 ICU-Bounces[0,age_group_80_plus]
   - H-Bounces[1,age_group_0_9] - H-Bounces[1,age_group_10_19]
   - H-Bounces[1,age_group_20_29] - H-Bounces[1,age_group_30_39]
   - H-Bounces[1,age_group_40_49] - H-Bounces[1,age_group_50_59]
   - H-Bounces[1,age_group_60_69] - H-Bounces[1,age_group_70_79]
   - H-Bounces[1,age_group_80_plus] + 1.58974e-06 m-tests[0,age_group_0_9]
   + 1.61623e-06 m-tests[0,age_group_10_19]
   + 2.22936e-06 m-tests[0,age_group_20_29]
   + 1.21699e-05 m-tests[0,age_group_30_39]
   + 1.97762e-05 m-tests[0,age_group_40_49]
   + 4.05127e-05 m-tests[0,age_group_50_59]
   + 5.48669e-05 m-tests[0,age_group_60_69]
   + 4.9738e-05 m-tests[0,age_group_70_79]
   + 1.4555e-05 m-tests[0,age_group_80_plus] <= 1.9998428863902789e+07
 R4: H-Bounces[0,age_group_0_9] <= 2.960983439684357
 R5: H-Bounces[0,age_group_10_19] <= 2.918288881199535
 R6: H-Bounces[0,age_group_20_29] <= 9.281002527281281
 R7: H-Bounces[0,age_group_30_39] <= 33.34053440028307
 R8: H-Bounces[0,age_group_40_49] <= 38.5012206565059
 R9: H-Bounces[0,age_group_50_59] <= 54.21513155796711
 R10: H-Bounces[0,age_group_60_69] <= 48.07328029515463
 R11: H-Bounces[0,age_group_70_79] <= 39.33480026418168
 R12: H-Bounces[0,age_group_80_plus] <= 64.02464376607905
 R13: H-Bounces[1,age_group_0_9] <= 2.958534760231826
 R14: H-Bounces[1,age_group_10_19] <= 2.910842758699157
 R15: H-Bounces[1,age_group_20_29] <= 9.263646756921602
 R16: H-Bounces[1,age_group_30_39] <= 33.37429400563178
 R17: H-Bounces[1,age_group_40_49] <= 38.42873845621118
 R18: H-Bounces[1,age_group_50_59] <= 54.12185288445627
 R19: H-Bounces[1,age_group_60_69] <= 48.0166264998989
 R20: H-Bounces[1,age_group_70_79] <= 39.24366576391524
 R21: H-Bounces[1,age_group_80_plus] <= 64.00555650116472
 R22: ICU-Bounces[0,age_group_0_9] <= 0.8449078709639167
 R23: ICU-Bounces[0,age_group_10_19] <= 0.8327251049181192
 R24: ICU-Bounces[0,age_group_20_29] <= 1.206005978121296
 R25: ICU-Bounces[0,age_group_30_39] <= 6.303382841432827
 R26: ICU-Bounces[0,age_group_40_49] <= 10.98620949324461
 R27: ICU-Bounces[0,age_group_50_59] <= 20.66764683701509
 R28: ICU-Bounces[0,age_group_60_69] <= 21.39677793483761
 R29: ICU-Bounces[0,age_group_70_79] <= 13.04176466815079
 R30: ICU-Bounces[0,age_group_80_plus] <= 3.798072087818249
 R31: ICU-Bounces[1,age_group_0_9] <= 0.8442091475211635
 R32: ICU-Bounces[1,age_group_10_19] <= 0.830600375875595
 R33: ICU-Bounces[1,age_group_20_29] <= 1.203750708526536
 R34: ICU-Bounces[1,age_group_30_39] <= 6.309765454096852
 R35: ICU-Bounces[1,age_group_40_49] <= 10.96552691166951
 R36: ICU-Bounces[1,age_group_50_59] <= 20.63208756368775
 R37: ICU-Bounces[1,age_group_60_69] <= 21.37156208377003
 R38: ICU-Bounces[1,age_group_70_79] <= 13.01154830254979
 R39: ICU-Bounces[1,age_group_80_plus] <= 3.796939792441975
 R40: m-tests[0,age_group_0_9] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_20_29] + m-tests[0,age_group_30_39]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_50_59]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_80_plus] <= 100000
 R41: m-tests[1,age_group_0_9] + m-tests[1,age_group_10_19]
   + m-tests[1,age_group_20_29] + m-tests[1,age_group_30_39]
   + m-tests[1,age_group_40_49] + m-tests[1,age_group_50_59]
   + m-tests[1,age_group_60_69] + m-tests[1,age_group_70_79]
   + m-tests[1,age_group_80_plus] <= 100000
 R42: a-tests[0,age_group_0_9] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_20_29] + a-tests[0,age_group_30_39]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_50_59]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_80_plus] <= 100000
 R43: a-tests[1,age_group_0_9] + a-tests[1,age_group_10_19]
   + a-tests[1,age_group_20_29] + a-tests[1,age_group_30_39]
   + a-tests[1,age_group_40_49] + a-tests[1,age_group_50_59]
   + a-tests[1,age_group_60_69] + a-tests[1,age_group_70_79]
   + a-tests[1,age_group_80_plus] <= 100000
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
 a-tests[1,age_group_0_9] <= 100000
 a-tests[1,age_group_10_19] <= 100000
 a-tests[1,age_group_20_29] <= 100000
 a-tests[1,age_group_30_39] <= 100000
 a-tests[1,age_group_40_49] <= 100000
 a-tests[1,age_group_50_59] <= 100000
 a-tests[1,age_group_60_69] <= 100000
 a-tests[1,age_group_70_79] <= 100000
 a-tests[1,age_group_80_plus] <= 100000
 m-tests[0,age_group_0_9] <= 100000
 m-tests[0,age_group_10_19] <= 100000
 m-tests[0,age_group_20_29] <= 100000
 m-tests[0,age_group_30_39] <= 100000
 m-tests[0,age_group_40_49] <= 100000
 m-tests[0,age_group_50_59] <= 100000
 m-tests[0,age_group_60_69] <= 100000
 m-tests[0,age_group_70_79] <= 100000
 m-tests[0,age_group_80_plus] <= 100000
 m-tests[1,age_group_0_9] <= 100000
 m-tests[1,age_group_10_19] <= 100000
 m-tests[1,age_group_20_29] <= 100000
 m-tests[1,age_group_30_39] <= 100000
 m-tests[1,age_group_40_49] <= 100000
 m-tests[1,age_group_50_59] <= 100000
 m-tests[1,age_group_60_69] <= 100000
 m-tests[1,age_group_70_79] <= 100000
 m-tests[1,age_group_80_plus] <= 100000
 Constant = 1
End
