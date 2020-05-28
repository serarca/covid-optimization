\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.0968220253740481 a-tests[1,age_group_20_29]
   + 0.2104571892597815 a-tests[1,age_group_30_39]
   + 0.0528081640802761 a-tests[1,age_group_60_69]
   + 0.2680324821243502 a-tests[1,age_group_40_49]
   + 0.2194001226053323 a-tests[1,age_group_50_59]
   - 3.46945e-18 m-tests[0,age_group_80_plus]
   - 0.1463978208260287 m-tests[0,age_group_20_29]
   - 0.7197810045403429 m-tests[0,age_group_30_39]
   - 0.2273345753424657 m-tests[0,age_group_60_69]
   - 1.075287689074117 m-tests[0,age_group_40_49]
   + 6.93889e-18 m-tests[0,age_group_0_9]
   - 0.9093383013698626 m-tests[0,age_group_50_59]
   - 0.1458570314058887 m-tests[1,age_group_20_29]
   - 0.5595120417025177 m-tests[1,age_group_30_39]
   - 0.1703878246449316 m-tests[1,age_group_60_69]
   - 0.8081340772204197 m-tests[1,age_group_40_49]
   - 0.6808045721927816 m-tests[1,age_group_50_59]
   + 1.5186402231261041e+09 Constant
Subject To
 HCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9999039635104656e+07
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
   - H-Bounces[1,age_group_50_59] + 0.0022656 m-tests[0,age_group_80_plus]
   + 3.9825e-05 m-tests[0,age_group_20_29]
   + 8.19975e-05 m-tests[0,age_group_30_39]
   + 1.167e-05 m-tests[0,age_group_10_19]
   + 0.00036849 m-tests[0,age_group_60_69]
   + 6.3647250000000008e-04 m-tests[0,age_group_70_79]
   + 9.9195e-05 m-tests[0,age_group_40_49]
   + 1.167e-05 m-tests[0,age_group_0_9]
   + 0.00019005 m-tests[0,age_group_50_59] <= 1.9998392615811128e+07
 HCAP[2]: - 0.92372234935164 H-Bounces[1,age_group_80_plus]
   - 0.92372234935164 H-Bounces[1,age_group_20_29]
   - 0.9237223493516399 H-Bounces[1,age_group_30_39]
   - 0.92372234935164 H-Bounces[1,age_group_10_19]
   - 0.92372234935164 H-Bounces[1,age_group_60_69]
   - 0.92372234935164 H-Bounces[1,age_group_70_79]
   - 0.92372234935164 H-Bounces[1,age_group_40_49]
   - 0.92372234935164 H-Bounces[1,age_group_0_9]
   - 0.9237223493516399 H-Bounces[1,age_group_50_59]
   - H-Bounces[2,age_group_80_plus] - H-Bounces[2,age_group_20_29]
   - H-Bounces[2,age_group_30_39] - H-Bounces[2,age_group_10_19]
   - H-Bounces[2,age_group_60_69] - H-Bounces[2,age_group_70_79]
   - H-Bounces[2,age_group_40_49] - H-Bounces[2,age_group_0_9]
   - H-Bounces[2,age_group_50_59] - 0.0005664 m-tests[0,age_group_80_plus]
   - 9.95625e-06 m-tests[0,age_group_20_29]
   - 2.04994e-05 m-tests[0,age_group_30_39]
   - 2.9175e-06 m-tests[0,age_group_10_19]
   - 9.21225e-05 m-tests[0,age_group_60_69]
   - 1.5911812499999996e-04 m-tests[0,age_group_70_79]
   - 2.47988e-05 m-tests[0,age_group_40_49]
   - 2.9175e-06 m-tests[0,age_group_0_9]
   - 4.75125e-05 m-tests[0,age_group_50_59]
   + 0.00169363296342416 m-tests[1,age_group_80_plus]
   + 2.98179e-05 m-tests[1,age_group_20_29]
   + 6.14892e-05 m-tests[1,age_group_30_39]
   + 8.74846e-06 m-tests[1,age_group_10_19]
   + 2.7618416340245298e-04 m-tests[1,age_group_60_69]
   + 4.7667619560717714e-04 m-tests[1,age_group_70_79]
   + 7.43411e-05 m-tests[1,age_group_40_49]
   + 8.749e-06 m-tests[1,age_group_0_9]
   + 1.4228687909694843e-04 m-tests[1,age_group_50_59]
   <= 1.9997932218803268e+07
 ICUCAP[0]: - H-Bounces[0,age_group_80_plus] - H-Bounces[0,age_group_20_29]
   - H-Bounces[0,age_group_30_39] - H-Bounces[0,age_group_10_19]
   - H-Bounces[0,age_group_60_69] - H-Bounces[0,age_group_70_79]
   - H-Bounces[0,age_group_40_49] - H-Bounces[0,age_group_0_9]
   - H-Bounces[0,age_group_50_59] <= 1.9999787006095346e+07
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
   - H-Bounces[1,age_group_50_59] + 0.0001344 m-tests[0,age_group_80_plus]
   + 5.175e-06 m-tests[0,age_group_20_29]
   + 1.55025e-05 m-tests[0,age_group_30_39]
   + 3.33e-06 m-tests[0,age_group_10_19]
   + 0.00016401 m-tests[0,age_group_60_69]
   + 2.1102750000000009e-04 m-tests[0,age_group_70_79]
   + 2.8305e-05 m-tests[0,age_group_40_49]
   + 3.33e-06 m-tests[0,age_group_0_9]
   + 7.245e-05 m-tests[0,age_group_50_59] <= 1.9999638429723207e+07
 ICUCAP[2]: - 0.9475616151022549 ICU-Bounces[1,age_group_80_plus]
   - 0.9475616151022548 ICU-Bounces[1,age_group_20_29]
   - 0.9475616151022549 ICU-Bounces[1,age_group_30_39]
   - 0.9475616151022548 ICU-Bounces[1,age_group_10_19]
   - 0.9475616151022548 ICU-Bounces[1,age_group_60_69]
   - 0.9475616151022549 ICU-Bounces[1,age_group_70_79]
   - 0.9475616151022549 ICU-Bounces[1,age_group_40_49]
   - 0.9475616151022548 ICU-Bounces[1,age_group_0_9]
   - 0.9475616151022549 ICU-Bounces[1,age_group_50_59]
   - H-Bounces[2,age_group_80_plus] - H-Bounces[2,age_group_20_29]
   - H-Bounces[2,age_group_30_39] - H-Bounces[2,age_group_10_19]
   - H-Bounces[2,age_group_60_69] - H-Bounces[2,age_group_70_79]
   - H-Bounces[2,age_group_40_49] - H-Bounces[2,age_group_0_9]
   - H-Bounces[2,age_group_50_59] - 3.36e-05 m-tests[0,age_group_80_plus]
   - 1.29375e-06 m-tests[0,age_group_20_29]
   - 3.87563e-06 m-tests[0,age_group_30_39]
   - 8.325e-07 m-tests[0,age_group_10_19]
   - 4.10025e-05 m-tests[0,age_group_60_69]
   - 5.27569e-05 m-tests[0,age_group_70_79]
   - 7.07625e-06 m-tests[0,age_group_40_49]
   - 8.325e-07 m-tests[0,age_group_0_9]
   - 1.81125e-05 m-tests[0,age_group_50_59]
   + 1.0046975206753497e-04 m-tests[1,age_group_80_plus]
   + 3.87465e-06 m-tests[1,age_group_20_29]
   + 1.16252e-05 m-tests[1,age_group_30_39]
   + 2.49635e-06 m-tests[1,age_group_10_19]
   + 1.2292589931785482e-04 m-tests[1,age_group_60_69]
   + 1.5804576924925053e-04 m-tests[1,age_group_70_79]
   + 2.1213e-05 m-tests[1,age_group_40_49]
   + 2.4965e-06 m-tests[1,age_group_0_9]
   + 5.4242e-05 m-tests[1,age_group_50_59] <= 1.9999526510645173e+07
 R6: H-Bounces[0,age_group_80_plus] <= 412.512896
 R7: H-Bounces[0,age_group_20_29] <= 21.929198175
 R8: H-Bounces[0,age_group_30_39] <= 48.04638046000001
 R9: H-Bounces[0,age_group_10_19] <= 6.01436401
 R10: H-Bounces[0,age_group_60_69] <= 144.67421003
 R11: H-Bounces[0,age_group_70_79] <= 168.232835515
 R12: H-Bounces[0,age_group_40_49] <= 55.27843071500001
 R13: H-Bounces[0,age_group_0_9] <= 6.20398984
 R14: H-Bounces[0,age_group_50_59] <= 97.4725906
 R15: H-Bounces[1,age_group_80_plus] <= 309.384672
 R16: H-Bounces[1,age_group_20_29] <= 16.44689863125
 R17: H-Bounces[1,age_group_30_39] <= 36.03478534500001
 R18: H-Bounces[1,age_group_10_19] <= 4.510773007500001
 R19: H-Bounces[1,age_group_60_69] <= 108.5056575225
 R20: H-Bounces[1,age_group_70_79] <= 126.17462663625
 R21: H-Bounces[1,age_group_40_49] <= 41.45882303625001
 R22: H-Bounces[1,age_group_0_9] <= 4.652992380000001
 R23: H-Bounces[1,age_group_50_59] <= 73.10444295000001
 R24: H-Bounces[2,age_group_80_plus]
   + 0.0005664 m-tests[0,age_group_80_plus] <= 240.9621506986579
 R25: H-Bounces[2,age_group_20_29] + 9.95625e-06 m-tests[0,age_group_20_29]
   <= 13.38243817018077
 R26: H-Bounces[2,age_group_30_39] + 2.04994e-05 m-tests[0,age_group_30_39]
   <= 30.82366148838643
 R27: H-Bounces[2,age_group_10_19] + 2.9175e-06 m-tests[0,age_group_10_19]
   <= 3.749826910098423
 R28: H-Bounces[2,age_group_60_69] + 9.21225e-05 m-tests[0,age_group_60_69]
   <= 87.81960722177615
 R29: H-Bounces[2,age_group_70_79]
   + 1.5911812499999999e-04 m-tests[0,age_group_70_79]
   <= 100.6519139519072
 R30: H-Bounces[2,age_group_40_49] + 2.47988e-05 m-tests[0,age_group_40_49]
   <= 35.92502476592674
 R31: H-Bounces[2,age_group_0_9] + 2.9175e-06 m-tests[0,age_group_0_9]
   <= 3.874063485636944
 R32: H-Bounces[2,age_group_50_59] + 4.75125e-05 m-tests[0,age_group_50_59]
   <= 61.7976677732273
 R33: ICU-Bounces[0,age_group_80_plus] <= 24.471104
 R34: ICU-Bounces[0,age_group_20_29] <= 2.849556825
 R35: ICU-Bounces[0,age_group_30_39] <= 9.08367954
 R36: ICU-Bounces[0,age_group_10_19] <= 1.71618099
 R37: ICU-Bounces[0,age_group_60_69] <= 64.39256746999999
 R38: ICU-Bounces[0,age_group_70_79] <= 55.77892948500001
 R39: ICU-Bounces[0,age_group_40_49] <= 15.773536785
 R40: ICU-Bounces[0,age_group_0_9] <= 1.77029016
 R41: ICU-Bounces[0,age_group_50_59] <= 37.15805940000001
 R42: ICU-Bounces[1,age_group_80_plus] <= 18.353328
 R43: ICU-Bounces[1,age_group_20_29] <= 2.13716761875
 R44: ICU-Bounces[1,age_group_30_39] <= 6.812759655000002
 R45: ICU-Bounces[1,age_group_10_19] <= 1.2871357425
 R46: ICU-Bounces[1,age_group_60_69] <= 48.29442560249999
 R47: ICU-Bounces[1,age_group_70_79] <= 41.83419711375
 R48: ICU-Bounces[1,age_group_40_49] <= 11.83015258875
 R49: ICU-Bounces[1,age_group_0_9] <= 1.32771762
 R50: ICU-Bounces[1,age_group_50_59] <= 27.86854455000001
 R51: ICU-Bounces[2,age_group_80_plus]
   + 3.36e-05 m-tests[0,age_group_80_plus] <= 14.29436487195428
 R52: ICU-Bounces[2,age_group_20_29]
   + 1.29375e-06 m-tests[0,age_group_20_29] <= 1.738960892170382
 R53: ICU-Bounces[2,age_group_30_39]
   + 3.87563e-06 m-tests[0,age_group_30_39] <= 5.827541232643807
 R54: ICU-Bounces[2,age_group_10_19] + 8.325e-07 m-tests[0,age_group_10_19]
   <= 1.070002023190038
 R55: ICU-Bounces[2,age_group_60_69]
   + 4.10025e-05 m-tests[0,age_group_60_69] <= 39.08733963050152
 R56: ICU-Bounces[2,age_group_70_79]
   + 5.27569e-05 m-tests[0,age_group_70_79] <= 33.37193951268294
 R57: ICU-Bounces[2,age_group_40_49]
   + 7.07625e-06 m-tests[0,age_group_40_49] <= 10.25109961187113
 R58: ICU-Bounces[2,age_group_0_9] + 8.325e-07 m-tests[0,age_group_0_9]
   <= 1.10545256273959
 R59: ICU-Bounces[2,age_group_50_59]
   + 1.81125e-05 m-tests[0,age_group_50_59] <= 23.55822694117505
 R60: m-tests[0,age_group_80_plus] + m-tests[0,age_group_20_29]
   + m-tests[0,age_group_30_39] + m-tests[0,age_group_10_19]
   + m-tests[0,age_group_60_69] + m-tests[0,age_group_70_79]
   + m-tests[0,age_group_40_49] + m-tests[0,age_group_0_9]
   + m-tests[0,age_group_50_59] <= 10000
 R61: m-tests[1,age_group_80_plus] + m-tests[1,age_group_20_29]
   + m-tests[1,age_group_30_39] + m-tests[1,age_group_10_19]
   + m-tests[1,age_group_60_69] + m-tests[1,age_group_70_79]
   + m-tests[1,age_group_40_49] + m-tests[1,age_group_0_9]
   + m-tests[1,age_group_50_59] <= 10000
 R62: m-tests[2,age_group_80_plus] + m-tests[2,age_group_20_29]
   + m-tests[2,age_group_30_39] + m-tests[2,age_group_10_19]
   + m-tests[2,age_group_60_69] + m-tests[2,age_group_70_79]
   + m-tests[2,age_group_40_49] + m-tests[2,age_group_0_9]
   + m-tests[2,age_group_50_59] <= 10000
 R63: a-tests[0,age_group_80_plus] + a-tests[0,age_group_20_29]
   + a-tests[0,age_group_30_39] + a-tests[0,age_group_10_19]
   + a-tests[0,age_group_60_69] + a-tests[0,age_group_70_79]
   + a-tests[0,age_group_40_49] + a-tests[0,age_group_0_9]
   + a-tests[0,age_group_50_59] <= 10000
 R64: a-tests[1,age_group_80_plus] + a-tests[1,age_group_20_29]
   + a-tests[1,age_group_30_39] + a-tests[1,age_group_10_19]
   + a-tests[1,age_group_60_69] + a-tests[1,age_group_70_79]
   + a-tests[1,age_group_40_49] + a-tests[1,age_group_0_9]
   + a-tests[1,age_group_50_59] <= 10000
 R65: a-tests[2,age_group_80_plus] + a-tests[2,age_group_20_29]
   + a-tests[2,age_group_30_39] + a-tests[2,age_group_10_19]
   + a-tests[2,age_group_60_69] + a-tests[2,age_group_70_79]
   + a-tests[2,age_group_40_49] + a-tests[2,age_group_0_9]
   + a-tests[2,age_group_50_59] <= 10000
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
 a-tests[2,age_group_80_plus] <= 10000
 a-tests[2,age_group_20_29] <= 10000
 a-tests[2,age_group_30_39] <= 10000
 a-tests[2,age_group_10_19] <= 10000
 a-tests[2,age_group_60_69] <= 10000
 a-tests[2,age_group_70_79] <= 10000
 a-tests[2,age_group_40_49] <= 10000
 a-tests[2,age_group_0_9] <= 10000
 a-tests[2,age_group_50_59] <= 10000
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
 m-tests[2,age_group_80_plus] <= 10000
 m-tests[2,age_group_20_29] <= 10000
 m-tests[2,age_group_30_39] <= 10000
 m-tests[2,age_group_10_19] <= 10000
 m-tests[2,age_group_60_69] <= 10000
 m-tests[2,age_group_70_79] <= 10000
 m-tests[2,age_group_40_49] <= 10000
 m-tests[2,age_group_0_9] <= 10000
 m-tests[2,age_group_50_59] <= 10000
 Constant = 1
End
