\ Model Linearization Heuristic
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 1e+06 u[2] - 1e+06 u[3] - 1e+06 u[12] - 1e+06 u[13] - 1e+06 u[22]
   - 1e+06 u[23] - 1e+06 u[32] - 1e+06 u[33] - 1e+06 u[42] - 1e+06 u[43]
   - 1e+06 u[52] - 1e+06 u[53] - 1e+06 u[62] - 1e+06 u[63] - 1e+06 u[72]
   - 1e+06 u[73] - 1e+06 u[82] - 1e+06 u[83]
Subject To
 H_capacity[t=19]: - u[2] - u[12] - u[22] - u[32] - u[42] - u[52] - u[62]
   - u[72] - u[82] <= 1.9989241250349946e+07
 BH_age_group_0_9[t=19]: u[2] <= 0.2623222219951093
 BH_age_group_10_19[t=19]: u[12] <= 0.2543043060480923
 BH_age_group_20_29[t=19]: u[22] <= 12.61224859504357
 BH_age_group_30_39[t=19]: u[32] <= 28.77882246464038
 BH_age_group_40_49[t=19]: u[42] <= 32.37540273087834
 BH_age_group_50_59[t=19]: u[52] <= 44.41506097266226
 BH_age_group_60_69[t=19]: u[62] <= 19.80842268780569
 BH_age_group_70_79[t=19]: u[72] <= 7.113359686745155
 BH_age_group_80_plus[t=19]: u[82] <= 17.44221094345916
 ICU_capacity[t=19]: - u[3] - u[13] - u[23] - u[33] - u[43] - u[53] - u[63]
   - u[73] - u[83] <= 384.9450808536108
 BICU_age_group_0_9[t=19]: u[3] <= 0.0748528705435916
 BICU_age_group_10_19[t=19]: u[13] <= 0.0725649819314608
 BICU_age_group_20_29[t=19]: u[23] <= 1.638879760937864
 BICU_age_group_30_39[t=19]: u[33] <= 5.440942653838075
 BICU_age_group_40_49[t=19]: u[43] <= 9.2382254579113
 BICU_age_group_50_59[t=19]: u[53] <= 16.93170832659501
 BICU_age_group_60_69[t=19]: u[63] <= 8.816465589370164
 BICU_age_group_70_79[t=19]: u[73] <= 2.358490761650525
 BICU_age_group_80_plus[t=19]: u[83] <= 1.034707428849272
 Mtest_cap[t=19]: u[0] + u[10] + u[20] + u[30] + u[40] + u[50] + u[60]
   + u[70] + u[80] <= 10000
 Atest_cap[t=19]: u[1] + u[11] + u[21] + u[31] + u[41] + u[51] + u[61]
   + u[71] + u[81] <= 10000
 lockdown_age_group_0_9_home[t=19]: u[4] <= 1
 lockdown_age_group_0_9_leisure[t=19]: u[5] <= 1
 lockdown_age_group_0_9_other[t=19]: u[6] <= 1
 lockdown_age_group_0_9_school[t=19]: u[7] <= 1
 lockdown_age_group_0_9_transport[t=19]: u[8] <= 1
 lockdown_age_group_0_9_work[t=19]: u[9] <= 1
 lockdown_age_group_10_19_home[t=19]: u[14] <= 1
 lockdown_age_group_10_19_leisure[t=19]: u[15] <= 1
 lockdown_age_group_10_19_other[t=19]: u[16] <= 1
 lockdown_age_group_10_19_school[t=19]: u[17] <= 1
 lockdown_age_group_10_19_transport[t=19]: u[18] <= 1
 lockdown_age_group_10_19_work[t=19]: u[19] <= 1
 lockdown_age_group_20_29_home[t=19]: u[24] <= 1
 lockdown_age_group_20_29_leisure[t=19]: u[25] <= 1
 lockdown_age_group_20_29_other[t=19]: u[26] <= 1
 lockdown_age_group_20_29_school[t=19]: u[27] <= 1
 lockdown_age_group_20_29_transport[t=19]: u[28] <= 1
 lockdown_age_group_20_29_work[t=19]: u[29] <= 1
 lockdown_age_group_30_39_home[t=19]: u[34] <= 1
 lockdown_age_group_30_39_leisure[t=19]: u[35] <= 1
 lockdown_age_group_30_39_other[t=19]: u[36] <= 1
 lockdown_age_group_30_39_school[t=19]: u[37] <= 1
 lockdown_age_group_30_39_transport[t=19]: u[38] <= 1
 lockdown_age_group_30_39_work[t=19]: u[39] <= 1
 lockdown_age_group_40_49_home[t=19]: u[44] <= 1
 lockdown_age_group_40_49_leisure[t=19]: u[45] <= 1
 lockdown_age_group_40_49_other[t=19]: u[46] <= 1
 lockdown_age_group_40_49_school[t=19]: u[47] <= 1
 lockdown_age_group_40_49_transport[t=19]: u[48] <= 1
 lockdown_age_group_40_49_work[t=19]: u[49] <= 1
 lockdown_age_group_50_59_home[t=19]: u[54] <= 1
 lockdown_age_group_50_59_leisure[t=19]: u[55] <= 1
 lockdown_age_group_50_59_other[t=19]: u[56] <= 1
 lockdown_age_group_50_59_school[t=19]: u[57] <= 1
 lockdown_age_group_50_59_transport[t=19]: u[58] <= 1
 lockdown_age_group_50_59_work[t=19]: u[59] <= 1
 lockdown_age_group_60_69_home[t=19]: u[64] <= 1
 lockdown_age_group_60_69_leisure[t=19]: u[65] <= 1
 lockdown_age_group_60_69_other[t=19]: u[66] <= 1
 lockdown_age_group_60_69_school[t=19]: u[67] <= 1
 lockdown_age_group_60_69_transport[t=19]: u[68] <= 1
 lockdown_age_group_60_69_work[t=19]: u[69] <= 1
 lockdown_age_group_70_79_home[t=19]: u[74] <= 1
 lockdown_age_group_70_79_leisure[t=19]: u[75] <= 1
 lockdown_age_group_70_79_other[t=19]: u[76] <= 1
 lockdown_age_group_70_79_school[t=19]: u[77] <= 1
 lockdown_age_group_70_79_transport[t=19]: u[78] <= 1
 lockdown_age_group_70_79_work[t=19]: u[79] <= 1
 lockdown_age_group_80_plus_home[t=19]: u[84] <= 1
 lockdown_age_group_80_plus_leisure[t=19]: u[85] <= 1
 lockdown_age_group_80_plus_other[t=19]: u[86] <= 1
 lockdown_age_group_80_plus_school[t=19]: u[87] <= 1
 lockdown_age_group_80_plus_transport[t=19]: u[88] <= 1
 lockdown_age_group_80_plus_work[t=19]: u[89] <= 1
Bounds
End
