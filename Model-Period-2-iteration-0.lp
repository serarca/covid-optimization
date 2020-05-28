\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  + 5.0699628609356868e+08 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9997937576327283e+07
 ICUCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9999527276394635e+07
 R2: H-Bounces[0,age_group_80_plus] <= 240.0097881571393
 R3: H-Bounces[0,age_group_20_29] <= 13.36110717882629
 R4: H-Bounces[0,age_group_30_39] <= 30.81880863801662
 R5: H-Bounces[0,age_group_10_19] <= 3.748235204923482
 R6: H-Bounces[0,age_group_60_69] <= 87.7506699617049
 R7: H-Bounces[0,age_group_70_79] <= 100.4895601373227
 R8: H-Bounces[0,age_group_40_49] <= 35.90045393533846
 R9: H-Bounces[0,age_group_0_9] <= 3.872638430420983
 R10: H-Bounces[0,age_group_50_59] <= 61.69571181836786
 R11: ICU-Bounces[0,age_group_80_plus] <= 14.23786878898284
 R12: ICU-Bounces[0,age_group_20_29] <= 1.736189068435054
 R13: ICU-Bounces[0,age_group_30_39] <= 5.82662374963691
 R14: ICU-Bounces[0,age_group_10_19] <= 1.069547834823924
 R15: ICU-Bounces[0,age_group_60_69] <= 39.05665657255074
 R16: ICU-Bounces[0,age_group_70_79] <= 33.3181098191656
 R17: ICU-Bounces[0,age_group_40_49] <= 10.24408839800146
 R18: ICU-Bounces[0,age_group_0_9] <= 1.105045927446604
 R19: ICU-Bounces[0,age_group_50_59] <= 23.51935975396344
 R20: m-tests[0,age_group_80_plus] + m-tests[0,age_group_20_29]
   + m-tests[0,age_group_30_39] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_0_9]
   + m-tests[0,age_group_50_59] <= 10000
 R21: a-tests[0,age_group_80_plus] + a-tests[0,age_group_20_29]
   + a-tests[0,age_group_30_39] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_0_9]
   + a-tests[0,age_group_50_59] <= 10000
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
 m-tests[0,age_group_80_plus] <= 10000
 m-tests[0,age_group_20_29] <= 10000
 m-tests[0,age_group_30_39] <= 10000
 m-tests[0,age_group_10_19] <= 10000
 m-tests[0,age_group_60_69] <= 10000
 m-tests[0,age_group_70_79] <= 10000
 m-tests[0,age_group_40_49] <= 10000
 m-tests[0,age_group_0_9] <= 10000
 m-tests[0,age_group_50_59] <= 10000
 Constant = 1
End
