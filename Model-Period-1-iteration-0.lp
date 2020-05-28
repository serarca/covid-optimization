\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.0968207696109936 a-tests[0,age_group_20_29]
   + 0.2104569058251967 a-tests[0,age_group_30_39]
   + 0.0528078285128487 a-tests[0,age_group_60_69]
   + 0.2680308935455785 a-tests[0,age_group_40_49]
   + 0.2193970624484436 a-tests[0,age_group_50_59]
   - 0.1461078012238334 m-tests[0,age_group_20_29]
   - 0.559611773014927 m-tests[0,age_group_30_39]
   - 0.1705312007949904 m-tests[0,age_group_60_69]
   - 0.8087683830757388 m-tests[0,age_group_40_49]
   - 0.6820634065754725 m-tests[0,age_group_50_59]
   + 1.0134414156525267e+09 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9998392615811128e+07
 HCAP[1]: - 0.92372234935164 H-Bounces[0,age_group_80_plus]
   - 0.92372234935164 H-Bounces[0,age_group_20_29]
   - 0.9237223493516399 H-Bounces[0,age_group_30_39]
   - 0.92372234935164 H-Bounces[0,age_group_10_19]
   - 0.92372234935164 H-Bounces[0,age_group_60_69]
   - 0.92372234935164 H-Bounces[0,age_group_70_79]
   - 0.92372234935164 H-Bounces[0,age_group_40_49]
   - 0.92372234935164 H-Bounces[0,age_group_0_9]
   - 0.9237223493516399 H-Bounces[0,age_group_50_59]
   - H-Bounces[1,age_group_80_plus] - H-Bounces[1,age_group_20_29]
   - H-Bounces[1,age_group_30_39] - H-Bounces[1,age_group_10_19]
   - H-Bounces[1,age_group_60_69] - H-Bounces[1,age_group_70_79]
   - H-Bounces[1,age_group_40_49] - H-Bounces[1,age_group_0_9]
   - H-Bounces[1,age_group_50_59]
   + 0.00170056044835869 m-tests[0,age_group_80_plus]
   + 2.98692e-05 m-tests[0,age_group_20_29]
   + 6.15001e-05 m-tests[0,age_group_30_39]
   + 8.75254e-06 m-tests[0,age_group_10_19]
   + 2.7641656394009935e-04 m-tests[0,age_group_60_69]
   + 4.7748926571756513e-04 m-tests[0,age_group_70_79]
   + 7.43994e-05 m-tests[0,age_group_40_49]
   + 8.75254e-06 m-tests[0,age_group_0_9]
   + 0.00014255 m-tests[0,age_group_50_59] <= 1.9997937576327283e+07
 ICUCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9999638429723207e+07
 ICUCAP[1]: - 0.9475616151022549 ICU-Bounces[0,age_group_80_plus]
   - 0.9475616151022548 ICU-Bounces[0,age_group_20_29]
   - 0.9475616151022549 ICU-Bounces[0,age_group_30_39]
   - 0.9475616151022548 ICU-Bounces[0,age_group_10_19]
   - 0.9475616151022548 ICU-Bounces[0,age_group_60_69]
   - 0.9475616151022549 ICU-Bounces[0,age_group_70_79]
   - 0.9475616151022549 ICU-Bounces[0,age_group_40_49]
   - 0.9475616151022548 ICU-Bounces[0,age_group_0_9]
   - 0.9475616151022549 ICU-Bounces[0,age_group_50_59]
   - H-Bounces[1,age_group_80_plus] - H-Bounces[1,age_group_20_29]
   - H-Bounces[1,age_group_30_39] - H-Bounces[1,age_group_10_19]
   - H-Bounces[1,age_group_60_69] - H-Bounces[1,age_group_70_79]
   - H-Bounces[1,age_group_40_49] - H-Bounces[1,age_group_0_9]
   - H-Bounces[1,age_group_50_59]
   + 1.0088070456365096e-04 m-tests[0,age_group_80_plus]
   + 3.88131e-06 m-tests[0,age_group_20_29]
   + 1.16273e-05 m-tests[0,age_group_30_39]
   + 2.49751e-06 m-tests[0,age_group_10_19]
   + 1.2302933770744307e-04 m-tests[0,age_group_60_69]
   + 1.5831534908611686e-04 m-tests[0,age_group_70_79]
   + 2.12297e-05 m-tests[0,age_group_40_49]
   + 2.49751e-06 m-tests[0,age_group_0_9]
   + 5.43423e-05 m-tests[0,age_group_50_59] <= 1.9999527276394635e+07
 R4: H-Bounces[0,age_group_80_plus] <= 309.384672
 R5: H-Bounces[0,age_group_20_29] <= 16.44689863125
 R6: H-Bounces[0,age_group_30_39] <= 36.03478534500001
 R7: H-Bounces[0,age_group_10_19] <= 4.510773007500001
 R8: H-Bounces[0,age_group_60_69] <= 108.5056575225
 R9: H-Bounces[0,age_group_70_79] <= 126.17462663625
 R10: H-Bounces[0,age_group_40_49] <= 41.45882303625001
 R11: H-Bounces[0,age_group_0_9] <= 4.652992380000001
 R12: H-Bounces[0,age_group_50_59] <= 73.10444295000001
 R13: H-Bounces[1,age_group_80_plus] <= 240.0097881571393
 R14: H-Bounces[1,age_group_20_29] <= 13.36110717882628
 R15: H-Bounces[1,age_group_30_39] <= 30.81880863801662
 R16: H-Bounces[1,age_group_10_19] <= 3.748235204923482
 R17: H-Bounces[1,age_group_60_69] <= 87.7506699617049
 R18: H-Bounces[1,age_group_70_79] <= 100.4895601373227
 R19: H-Bounces[1,age_group_40_49] <= 35.90045393533845
 R20: H-Bounces[1,age_group_0_9] <= 3.872638430420983
 R21: H-Bounces[1,age_group_50_59] <= 61.69571181836786
 R22: ICU-Bounces[0,age_group_80_plus] <= 18.353328
 R23: ICU-Bounces[0,age_group_20_29] <= 2.13716761875
 R24: ICU-Bounces[0,age_group_30_39] <= 6.812759655000002
 R25: ICU-Bounces[0,age_group_10_19] <= 1.2871357425
 R26: ICU-Bounces[0,age_group_60_69] <= 48.29442560249999
 R27: ICU-Bounces[0,age_group_70_79] <= 41.83419711375
 R28: ICU-Bounces[0,age_group_40_49] <= 11.83015258875
 R29: ICU-Bounces[0,age_group_0_9] <= 1.32771762
 R30: ICU-Bounces[0,age_group_50_59] <= 27.86854455000001
 R31: ICU-Bounces[1,age_group_80_plus] <= 14.23786878898284
 R32: ICU-Bounces[1,age_group_20_29] <= 1.736189068435054
 R33: ICU-Bounces[1,age_group_30_39] <= 5.82662374963691
 R34: ICU-Bounces[1,age_group_10_19] <= 1.069547834823924
 R35: ICU-Bounces[1,age_group_60_69] <= 39.05665657255074
 R36: ICU-Bounces[1,age_group_70_79] <= 33.31810981916559
 R37: ICU-Bounces[1,age_group_40_49] <= 10.24408839800146
 R38: ICU-Bounces[1,age_group_0_9] <= 1.105045927446604
 R39: ICU-Bounces[1,age_group_50_59] <= 23.51935975396344
 R40: m-tests[0,age_group_80_plus] + m-tests[0,age_group_20_29]
   + m-tests[0,age_group_30_39] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_0_9]
   + m-tests[0,age_group_50_59] <= 10000
 R41: m-tests[1,age_group_80_plus] + m-tests[1,age_group_20_29]
   + m-tests[1,age_group_30_39] + m-tests[1,age_group_10_19]
   + m-tests[1,age_group_60_69] + m-tests[1,age_group_70_79]
   + m-tests[1,age_group_40_49] + m-tests[1,age_group_0_9]
   + m-tests[1,age_group_50_59] <= 10000
 R42: a-tests[0,age_group_80_plus] + a-tests[0,age_group_20_29]
   + a-tests[0,age_group_30_39] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_0_9]
   + a-tests[0,age_group_50_59] <= 10000
 R43: a-tests[1,age_group_80_plus] + a-tests[1,age_group_20_29]
   + a-tests[1,age_group_30_39] + a-tests[1,age_group_10_19]
   + a-tests[1,age_group_60_69] + a-tests[1,age_group_70_79]
   + a-tests[1,age_group_40_49] + a-tests[1,age_group_0_9]
   + a-tests[1,age_group_50_59] <= 10000
Bounds
 a-tests[0,age_group_80_plus] <= 10000
 a-tests[0,age_group_20_29] <= 10000
 a-tests[0,age_group_30_39] <= 10000
 a-tests[0,age_group_10_19] <= 10000
 a-tests[0,age_group_60_69] <= 10000
 a-tests[0,age_group_70_79] <= 10000
 a-tests[0,age_group_40_49] <= 10000
 a-tests[0,age_group_0_9] <= 10000
 a-tests[0,age_group_50_59] <= 10000
 a-tests[1,age_group_80_plus] <= 10000
 a-tests[1,age_group_20_29] <= 10000
 a-tests[1,age_group_30_39] <= 10000
 a-tests[1,age_group_10_19] <= 10000
 a-tests[1,age_group_60_69] <= 10000
 a-tests[1,age_group_70_79] <= 10000
 a-tests[1,age_group_40_49] <= 10000
 a-tests[1,age_group_0_9] <= 10000
 a-tests[1,age_group_50_59] <= 10000
 m-tests[0,age_group_80_plus] <= 10000
 m-tests[0,age_group_20_29] <= 10000
 m-tests[0,age_group_30_39] <= 10000
 m-tests[0,age_group_10_19] <= 10000
 m-tests[0,age_group_60_69] <= 10000
 m-tests[0,age_group_70_79] <= 10000
 m-tests[0,age_group_40_49] <= 10000
 m-tests[0,age_group_0_9] <= 10000
 m-tests[0,age_group_50_59] <= 10000
 m-tests[1,age_group_80_plus] <= 10000
 m-tests[1,age_group_20_29] <= 10000
 m-tests[1,age_group_30_39] <= 10000
 m-tests[1,age_group_10_19] <= 10000
 m-tests[1,age_group_60_69] <= 10000
 m-tests[1,age_group_70_79] <= 10000
 m-tests[1,age_group_40_49] <= 10000
 m-tests[1,age_group_0_9] <= 10000
 m-tests[1,age_group_50_59] <= 10000
 Constant = 1
End
