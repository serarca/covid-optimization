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
   - 99968.53696906136 ICU-Bounces[1,age_group_0_9]
   - 99968.53696906136 ICU-Bounces[1,age_group_10_19]
   - 99942.31777661249 ICU-Bounces[1,age_group_20_29]
   - 99900.36706869428 ICU-Bounces[1,age_group_30_39]
   - 99826.95332983744 ICU-Bounces[1,age_group_40_49]
   - 99659.15049816466 ICU-Bounces[1,age_group_50_59]
   - 99339.2763502884 ICU-Bounces[1,age_group_60_69]
   - 98898.79391714736 ICU-Bounces[1,age_group_70_79]
   - 98342.94703723125 ICU-Bounces[1,age_group_80_plus]
   - 100000 ICU-Bounces[2,age_group_0_9]
   - 100000 ICU-Bounces[2,age_group_10_19]
   - 100000 ICU-Bounces[2,age_group_20_29]
   - 100000 ICU-Bounces[2,age_group_30_39]
   - 100000 ICU-Bounces[2,age_group_40_49]
   - 100000 ICU-Bounces[2,age_group_50_59]
   - 100000 ICU-Bounces[2,age_group_60_69]
   - 100000 ICU-Bounces[2,age_group_70_79]
   - 100000 ICU-Bounces[2,age_group_80_plus]
   - 99954.23340961098 H-Bounces[0,age_group_0_9]
   - 99954.23340961098 H-Bounces[0,age_group_10_19]
   - 99916.09458428681 H-Bounces[0,age_group_20_29]
   - 99855.07246376811 H-Bounces[0,age_group_30_39]
   - 99748.28375286041 H-Bounces[0,age_group_40_49]
   - 99504.19527078565 H-Bounces[0,age_group_50_59]
   - 99038.90160183066 H-Bounces[0,age_group_60_69]
   - 98398.16933638444 H-Bounces[0,age_group_70_79]
   - 97589.62623951182 H-Bounces[0,age_group_80_plus]
   - 99954.23340961098 H-Bounces[1,age_group_0_9]
   - 99954.23340961098 H-Bounces[1,age_group_10_19]
   - 99916.09458428681 H-Bounces[1,age_group_20_29]
   - 99855.07246376811 H-Bounces[1,age_group_30_39]
   - 99748.28375286041 H-Bounces[1,age_group_40_49]
   - 99504.19527078565 H-Bounces[1,age_group_50_59]
   - 99038.90160183066 H-Bounces[1,age_group_60_69]
   - 98398.16933638444 H-Bounces[1,age_group_70_79]
   - 97589.62623951182 H-Bounces[1,age_group_80_plus]
   - 100000 H-Bounces[2,age_group_0_9]
   - 100000 H-Bounces[2,age_group_10_19]
   - 100000 H-Bounces[2,age_group_20_29]
   - 100000 H-Bounces[2,age_group_30_39]
   - 100000 H-Bounces[2,age_group_40_49]
   - 100000 H-Bounces[2,age_group_50_59]
   - 100000 H-Bounces[2,age_group_60_69]
   - 100000 H-Bounces[2,age_group_70_79]
   - 100000 H-Bounces[2,age_group_80_plus]
   + 0.6776450454481298 a-tests[0,age_group_20_29]
   + 0.6376771863692809 a-tests[0,age_group_30_39]
   + 0.2213163890364314 a-tests[0,age_group_40_49]
   + 0.6956744504696684 a-tests[1,age_group_20_29]
   + 0.6499590233799752 a-tests[1,age_group_30_39]
   + 0.2231382298035314 a-tests[1,age_group_40_49]
   + 4.06576e-20 m-tests[0,age_group_0_9]
   + 1.0842e-18 m-tests[0,age_group_10_19]
   + 0.022309388811318 m-tests[0,age_group_20_29]
   + 0.020644734503307 m-tests[0,age_group_30_39]
   + 0.00211664047582355 m-tests[0,age_group_40_49]
   - 1.38778e-17 m-tests[0,age_group_50_59]
   + 1.38778e-17 m-tests[0,age_group_60_69]
   - 2.77556e-17 m-tests[0,age_group_70_79]
   + 1.5194925738389482e+09 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.9995967842141904e+07
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
   - H-Bounces[1,age_group_80_plus] + 5.57611e-06 m-tests[0,age_group_0_9]
   + 5.67901e-06 m-tests[0,age_group_10_19]
   + 1.83532e-05 m-tests[0,age_group_20_29]
   + 6.42587e-05 m-tests[0,age_group_30_39]
   + 6.94238e-05 m-tests[0,age_group_40_49]
   + 1.0644158383468643e-04 m-tests[0,age_group_50_59]
   + 1.2341427458589284e-04 m-tests[0,age_group_60_69]
   + 1.5036617168946126e-04 m-tests[0,age_group_70_79]
   + 2.0639425083660815e-04 m-tests[0,age_group_80_plus]
   <= 1.9995997138509233e+07
 HCAP[2]: - 0.92372234935164 H-Bounces[1,age_group_0_9]
   - 0.92372234935164 H-Bounces[1,age_group_10_19]
   - 0.92372234935164 H-Bounces[1,age_group_20_29]
   - 0.9237223493516399 H-Bounces[1,age_group_30_39]
   - 0.92372234935164 H-Bounces[1,age_group_40_49]
   - 0.9237223493516399 H-Bounces[1,age_group_50_59]
   - 0.92372234935164 H-Bounces[1,age_group_60_69]
   - 0.92372234935164 H-Bounces[1,age_group_70_79]
   - 0.92372234935164 H-Bounces[1,age_group_80_plus]
   - H-Bounces[2,age_group_0_9] - H-Bounces[2,age_group_10_19]
   - H-Bounces[2,age_group_20_29] - H-Bounces[2,age_group_30_39]
   - H-Bounces[2,age_group_40_49] - H-Bounces[2,age_group_50_59]
   - H-Bounces[2,age_group_60_69] - H-Bounces[2,age_group_70_79]
   - H-Bounces[2,age_group_80_plus] - 1.39403e-06 m-tests[0,age_group_0_9]
   - 1.41975e-06 m-tests[0,age_group_10_19]
   - 4.5883e-06 m-tests[0,age_group_20_29]
   - 1.60647e-05 m-tests[0,age_group_30_39]
   - 1.7356e-05 m-tests[0,age_group_40_49]
   - 2.66104e-05 m-tests[0,age_group_50_59]
   - 3.08536e-05 m-tests[0,age_group_60_69]
   - 3.75915e-05 m-tests[0,age_group_70_79]
   - 5.15986e-05 m-tests[0,age_group_80_plus]
   + 5.49569e-06 m-tests[1,age_group_0_9]
   + 5.66639e-06 m-tests[1,age_group_10_19]
   + 1.82153e-05 m-tests[1,age_group_20_29]
   + 6.34965e-05 m-tests[1,age_group_30_39]
   + 6.92367e-05 m-tests[1,age_group_40_49]
   + 0.000106387 m-tests[1,age_group_50_59]
   + 1.212766151022353e-04 m-tests[1,age_group_60_69]
   + 0.00014672 m-tests[1,age_group_70_79]
   + 2.4453539172171387e-04 m-tests[1,age_group_80_plus]
   <= 1.9996014203202963e+07
 ICUCAP[0]: - H-Bounces[0,age_group_0_9] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_20_29] - H-Bounces[0,age_group_30_39]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_50_59]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_80_plus] <= 1.999841992402254e+07
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
   - H-Bounces[1,age_group_80_plus] + 1.59113e-06 m-tests[0,age_group_0_9]
   + 1.62049e-06 m-tests[0,age_group_10_19]
   + 2.38488e-06 m-tests[0,age_group_20_29]
   + 1.21488e-05 m-tests[0,age_group_30_39]
   + 1.98099e-05 m-tests[0,age_group_40_49]
   + 4.05772e-05 m-tests[0,age_group_50_59]
   + 5.49301e-05 m-tests[0,age_group_60_69]
   + 4.98551e-05 m-tests[0,age_group_70_79]
   + 1.22437e-05 m-tests[0,age_group_80_plus] <= 1.9998424673457623e+07
 ICUCAP[2]: - 0.9475616151022548 ICU-Bounces[1,age_group_0_9]
   - 0.9475616151022548 ICU-Bounces[1,age_group_10_19]
   - 0.9475616151022548 ICU-Bounces[1,age_group_20_29]
   - 0.9475616151022549 ICU-Bounces[1,age_group_30_39]
   - 0.9475616151022549 ICU-Bounces[1,age_group_40_49]
   - 0.9475616151022549 ICU-Bounces[1,age_group_50_59]
   - 0.9475616151022548 ICU-Bounces[1,age_group_60_69]
   - 0.9475616151022549 ICU-Bounces[1,age_group_70_79]
   - 0.9475616151022549 ICU-Bounces[1,age_group_80_plus]
   - H-Bounces[2,age_group_0_9] - H-Bounces[2,age_group_10_19]
   - H-Bounces[2,age_group_20_29] - H-Bounces[2,age_group_30_39]
   - H-Bounces[2,age_group_40_49] - H-Bounces[2,age_group_50_59]
   - H-Bounces[2,age_group_60_69] - H-Bounces[2,age_group_70_79]
   - H-Bounces[2,age_group_80_plus] - 3.97781e-07 m-tests[0,age_group_0_9]
   - 4.05122e-07 m-tests[0,age_group_10_19]
   - 5.96219e-07 m-tests[0,age_group_20_29]
   - 3.0372e-06 m-tests[0,age_group_30_39]
   - 4.95247e-06 m-tests[0,age_group_40_49]
   - 1.01443e-05 m-tests[0,age_group_50_59]
   - 1.37325e-05 m-tests[0,age_group_60_69]
   - 1.24638e-05 m-tests[0,age_group_70_79]
   - 3.06093e-06 m-tests[0,age_group_80_plus]
   + 1.56818e-06 m-tests[1,age_group_0_9]
   + 1.61689e-06 m-tests[1,age_group_10_19]
   + 2.36696e-06 m-tests[1,age_group_20_29]
   + 1.20047e-05 m-tests[1,age_group_30_39]
   + 1.97565e-05 m-tests[1,age_group_40_49]
   + 4.05564e-05 m-tests[1,age_group_50_59]
   + 5.39786e-05 m-tests[1,age_group_60_69]
   + 4.86462e-05 m-tests[1,age_group_70_79]
   + 1.45063e-05 m-tests[1,age_group_80_plus] <= 1.9998426847607143e+07
 R6: H-Bounces[0,age_group_0_9] <= 2.963569225637573
 R7: H-Bounces[0,age_group_10_19] <= 2.925986005083731
 R8: H-Bounces[0,age_group_20_29] <= 9.298163466128351
 R9: H-Bounces[0,age_group_30_39] <= 33.28343965133264
 R10: H-Bounces[0,age_group_40_49] <= 38.56787993192526
 R11: H-Bounces[0,age_group_50_59] <= 54.30404878109042
 R12: H-Bounces[0,age_group_60_69] <= 48.13151491115468
 R13: H-Bounces[0,age_group_70_79] <= 39.43005630802973
 R14: H-Bounces[0,age_group_80_plus] <= 64.04641645369117
 R15: H-Bounces[1,age_group_0_9] <= 2.960983439684357
 R16: H-Bounces[1,age_group_10_19] <= 2.918288881199535
 R17: H-Bounces[1,age_group_20_29] <= 9.281002527281281
 R18: H-Bounces[1,age_group_30_39] <= 33.34053440028307
 R19: H-Bounces[1,age_group_40_49] <= 38.5012206565059
 R20: H-Bounces[1,age_group_50_59] <= 54.21513155796711
 R21: H-Bounces[1,age_group_60_69] <= 48.07328029515463
 R22: H-Bounces[1,age_group_70_79] <= 39.33480026418168
 R23: H-Bounces[1,age_group_80_plus] <= 64.02464376607905
 R24: H-Bounces[2,age_group_0_9] + 1.39403e-06 m-tests[0,age_group_0_9]
   <= 2.992168993697004
 R25: H-Bounces[2,age_group_10_19] + 1.41975e-06 m-tests[0,age_group_10_19]
   <= 2.913736651667281
 R26: H-Bounces[2,age_group_20_29] + 4.5883e-06 m-tests[0,age_group_20_29]
   <= 9.214607000852528
 R27: H-Bounces[2,age_group_30_39] + 1.60647e-05 m-tests[0,age_group_30_39]
   <= 33.4087446332789
 R28: H-Bounces[2,age_group_40_49] + 1.7356e-05 m-tests[0,age_group_40_49]
   <= 38.30766124736665
 R29: H-Bounces[2,age_group_50_59] + 2.66104e-05 m-tests[0,age_group_50_59]
   <= 54.00752170062188
 R30: H-Bounces[2,age_group_60_69] + 3.08536e-05 m-tests[0,age_group_60_69]
   <= 48.57779238054235
 R31: H-Bounces[2,age_group_70_79] + 3.75915e-05 m-tests[0,age_group_70_79]
   <= 39.89844222262363
 R32: H-Bounces[2,age_group_80_plus]
   + 5.15986e-05 m-tests[0,age_group_80_plus] <= 64.11594902186791
 R33: ICU-Bounces[0,age_group_0_9] <= 0.8456457173413126
 R34: ICU-Bounces[0,age_group_10_19] <= 0.834921456463481
 R35: ICU-Bounces[0,age_group_20_29] <= 1.208235930626848
 R36: ICU-Bounces[0,age_group_30_39] <= 6.292588471536136
 R37: ICU-Bounces[0,age_group_40_49] <= 11.00523052042083
 R38: ICU-Bounces[0,age_group_50_59] <= 20.70154345798475
 R39: ICU-Bounces[0,age_group_60_69] <= 21.42269738820179
 R40: ICU-Bounces[0,age_group_70_79] <= 13.0733475641803
 R41: ICU-Bounces[0,age_group_80_plus] <= 3.799363687930832
 R42: ICU-Bounces[1,age_group_0_9] <= 0.8449078709639167
 R43: ICU-Bounces[1,age_group_10_19] <= 0.8327251049181192
 R44: ICU-Bounces[1,age_group_20_29] <= 1.206005978121296
 R45: ICU-Bounces[1,age_group_30_39] <= 6.303382841432827
 R46: ICU-Bounces[1,age_group_40_49] <= 10.98620949324461
 R47: ICU-Bounces[1,age_group_50_59] <= 20.66764683701509
 R48: ICU-Bounces[1,age_group_60_69] <= 21.39677793483761
 R49: ICU-Bounces[1,age_group_70_79] <= 13.04176466815079
 R50: ICU-Bounces[1,age_group_80_plus] <= 3.798072087818249
 R51: ICU-Bounces[2,age_group_0_9] + 3.97781e-07 m-tests[0,age_group_0_9]
   <= 0.8538065766076282
 R52: ICU-Bounces[2,age_group_10_19]
   + 4.05122e-07 m-tests[0,age_group_10_19] <= 0.8314261396788385
 R53: ICU-Bounces[2,age_group_20_29]
   + 5.96219e-07 m-tests[0,age_group_20_29] <= 1.197378310845244
 R54: ICU-Bounces[2,age_group_30_39]
   + 3.0372e-06 m-tests[0,age_group_30_39] <= 6.316278711880314
 R55: ICU-Bounces[2,age_group_40_49]
   + 4.95247e-06 m-tests[0,age_group_40_49] <= 10.93097788806606
 R56: ICU-Bounces[2,age_group_50_59]
   + 1.01443e-05 m-tests[0,age_group_50_59] <= 20.58850274775089
 R57: ICU-Bounces[2,age_group_60_69]
   + 1.37325e-05 m-tests[0,age_group_60_69] <= 21.62132955665758
 R58: ICU-Bounces[2,age_group_70_79]
   + 1.24638e-05 m-tests[0,age_group_70_79] <= 13.22864462507761
 R59: ICU-Bounces[2,age_group_80_plus]
   + 3.06093e-06 m-tests[0,age_group_80_plus] <= 3.803488501297249
 R60: m-tests[0,age_group_0_9] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_20_29] + m-tests[0,age_group_30_39]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_50_59]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_80_plus] <= 100000
 R61: m-tests[1,age_group_0_9] + m-tests[1,age_group_10_19]
   + m-tests[1,age_group_20_29] + m-tests[1,age_group_30_39]
   + m-tests[1,age_group_40_49] + m-tests[1,age_group_50_59]
   + m-tests[1,age_group_60_69] + m-tests[1,age_group_70_79]
   + m-tests[1,age_group_80_plus] <= 100000
 R62: m-tests[2,age_group_0_9] + m-tests[2,age_group_10_19]
   + m-tests[2,age_group_20_29] + m-tests[2,age_group_30_39]
   + m-tests[2,age_group_40_49] + m-tests[2,age_group_50_59]
   + m-tests[2,age_group_60_69] + m-tests[2,age_group_70_79]
   + m-tests[2,age_group_80_plus] <= 100000
 R63: a-tests[0,age_group_0_9] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_20_29] + a-tests[0,age_group_30_39]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_50_59]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_80_plus] <= 100000
 R64: a-tests[1,age_group_0_9] + a-tests[1,age_group_10_19]
   + a-tests[1,age_group_20_29] + a-tests[1,age_group_30_39]
   + a-tests[1,age_group_40_49] + a-tests[1,age_group_50_59]
   + a-tests[1,age_group_60_69] + a-tests[1,age_group_70_79]
   + a-tests[1,age_group_80_plus] <= 100000
 R65: a-tests[2,age_group_0_9] + a-tests[2,age_group_10_19]
   + a-tests[2,age_group_20_29] + a-tests[2,age_group_30_39]
   + a-tests[2,age_group_40_49] + a-tests[2,age_group_50_59]
   + a-tests[2,age_group_60_69] + a-tests[2,age_group_70_79]
   + a-tests[2,age_group_80_plus] <= 100000
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
 a-tests[2,age_group_0_9] <= 100000
 a-tests[2,age_group_10_19] <= 100000
 a-tests[2,age_group_20_29] <= 100000
 a-tests[2,age_group_30_39] <= 100000
 a-tests[2,age_group_40_49] <= 100000
 a-tests[2,age_group_50_59] <= 100000
 a-tests[2,age_group_60_69] <= 100000
 a-tests[2,age_group_70_79] <= 100000
 a-tests[2,age_group_80_plus] <= 100000
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
 m-tests[2,age_group_0_9] <= 100000
 m-tests[2,age_group_10_19] <= 100000
 m-tests[2,age_group_20_29] <= 100000
 m-tests[2,age_group_30_39] <= 100000
 m-tests[2,age_group_40_49] <= 100000
 m-tests[2,age_group_50_59] <= 100000
 m-tests[2,age_group_60_69] <= 100000
 m-tests[2,age_group_70_79] <= 100000
 m-tests[2,age_group_80_plus] <= 100000
 Constant = 1
End
