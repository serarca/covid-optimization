\ Model Linearization Heuristic V2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.7366276307942912 u_vars_time_1[0] - 333493.0821925869 u_vars_time_1[2]
   - 333582.1787987602 u_vars_time_1[3] + 9.09495e-13 u_vars_time_1[4]
   + 186635.1771658132 u_vars_time_1[5] + -0 u_vars_time_1[6]
   + 1.646680551118088e+07 u_vars_time_1[7]
   + 222451.0889714831 u_vars_time_1[8]
   - 2.224578115486866 u_vars_time_1[10]
   - 438361.6326656485 u_vars_time_1[12]
   - 438478.634069191 u_vars_time_1[13] + 2.18279e-11 u_vars_time_1[14]
   + 186635.1771658132 u_vars_time_1[15] + -0 u_vars_time_1[16]
   + 1.9418732226776589e+07 u_vars_time_1[17]
   + 222451.0889714831 u_vars_time_1[18]
   + 5.8047738935645688e+06 u_vars_time_1[19]
   - 0.4607391889289885 u_vars_time_2[0]
   - 333633.4095054139 u_vars_time_2[2] - 333681.152808971 u_vars_time_2[3]
   + -0 u_vars_time_2[4] + 186159.6834988012 u_vars_time_2[5]
   + -0 u_vars_time_2[6] + 1.6466469863246087e+07 u_vars_time_2[7]
   + 221884.3464868555 u_vars_time_2[8]
   - 1.395876887401507 u_vars_time_2[10]
   - 438545.7281610357 u_vars_time_2[12]
   - 438608.4845314313 u_vars_time_2[13] + -0 u_vars_time_2[14]
   + 186159.6834988012 u_vars_time_2[15] + -0 u_vars_time_2[16]
   + 1.9369258787013244e+07 u_vars_time_2[17]
   + 221884.3464868555 u_vars_time_2[18]
   + 5.7899849707755139e+06 u_vars_time_2[19]
   - 333786.1720555538 u_vars_time_3[2]
   - 333786.1720555538 u_vars_time_3[3] + -0 u_vars_time_3[4]
   + 185876.2950803121 u_vars_time_3[5] + -0 u_vars_time_3[6]
   + 1.6466102479173712e+07 u_vars_time_3[7]
   + 221546.5748874597 u_vars_time_3[8]
   - 438746.5274871168 u_vars_time_3[12]
   - 438746.5274871168 u_vars_time_3[13] + -0 u_vars_time_3[14]
   + 185876.2950803121 u_vars_time_3[15] + -0 u_vars_time_3[16]
   + 1.9339773221117374e+07 u_vars_time_3[17]
   + 221546.5748874597 u_vars_time_3[18]
   + 5.7811709534055591e+06 u_vars_time_3[19]
   + 1.3401354171916045e+08 Constant
Subject To
 Const0_time_1: u_vars_time_1[2] <= 25.29054794206607
 Const1_time_1: u_vars_time_1[12] <= 42.36445113880747
 Const2_time_1: - u_vars_time_1[3] - u_vars_time_1[13]
   <= 2429.276904932042
 Const3_time_1: u_vars_time_1[3] <= 7.216583088867182
 Const4_time_1: u_vars_time_1[13] <= 12.08857089050804
 Const5_time_1: u_vars_time_1[0] + u_vars_time_1[10] <= 30000
 Const6_time_1: u_vars_time_1[1] + u_vars_time_1[11] <= 30000
 R7: - u_vars_time_1[8] + 0.25 u_vars_time_1[9] <= 0
 R8: - u_vars_time_1[18] + 0.25 u_vars_time_1[19] <= 0
 Const0_time_2: u_vars_time_2[2] <= 27.68181636881371
 Const1_time_2: u_vars_time_2[12] <= 46.98150772693064
 Const2_time_2: - 0.9475616151022548 u_vars_time_1[3]
   - 0.9475616151022548 u_vars_time_1[13] - u_vars_time_2[3]
   - u_vars_time_2[13] <= 2411.680551026998
 Const3_time_2: u_vars_time_2[3] <= 7.898924465137074
 Const4_time_2: u_vars_time_2[13] <= 13.40603433853291
 Const5_time_2: u_vars_time_2[0] + u_vars_time_2[10] <= 30000
 Const6_time_2: u_vars_time_2[1] + u_vars_time_2[11] <= 30000
 R16: - u_vars_time_2[8] + 0.25 u_vars_time_2[9] <= 0
 R17: - u_vars_time_2[18] + 0.25 u_vars_time_2[19] <= 0
 Const0_time_3: - 0.6318434032379874 u_vars_time_1[4]
   - 0.5264721181147958 u_vars_time_1[5]
   - 0.6879958196880134 u_vars_time_1[7]
   - 0.0386885315560868 u_vars_time_1[8]
   - 0.1260944131604518 u_vars_time_1[14]
   - 0.0804013967439047 u_vars_time_1[15]
   - 0.0444675402668042 u_vars_time_1[17]
   - 0.0097670857396889 u_vars_time_1[18] + u_vars_time_3[2]
   <= 26.58154789963915
 Const1_time_3: - 0.0588881579244574 u_vars_time_1[4]
   - 0.0375487706148193 u_vars_time_1[5]
   - 0.020767070156242 u_vars_time_1[7]
   - 0.00456138912545303 u_vars_time_1[8]
   - 0.8634255241824559 u_vars_time_1[14]
   - 1.562564760737781 u_vars_time_1[15]
   - 2.153883870166963 u_vars_time_1[17]
   - 0.2744499284707828 u_vars_time_1[18]
   - 0.00375894269299986 u_vars_time_1[19] + u_vars_time_3[12]
   <= 44.98230835345944
 Const2_time_3: - 0.8978730144151937 u_vars_time_1[3]
   + 0.1970982089692323 u_vars_time_1[4]
   + 0.1609416931850573 u_vars_time_1[5]
   + 0.2022433953026024 u_vars_time_1[7]
   + 0.0123412370068147 u_vars_time_1[8]
   - 0.8978730144151937 u_vars_time_1[13]
   + 0.2823565888047885 u_vars_time_1[14]
   + 0.4688155359395041 u_vars_time_1[15]
   + 0.6272930759849566 u_vars_time_1[17]
   + 0.0811004847747104 u_vars_time_1[18]
   + 0.00107260318489199 u_vars_time_1[19]
   - 0.9475616151022548 u_vars_time_2[3]
   - 0.9475616151022548 u_vars_time_2[13] - u_vars_time_3[3]
   - u_vars_time_3[13] <= 2395.891345488667
 Const3_time_3: - 0.1802946471964437 u_vars_time_1[4]
   - 0.1502272624954816 u_vars_time_1[5]
   - 0.1963175732271709 u_vars_time_1[7]
   - 0.0110396581046931 u_vars_time_1[8]
   - 0.0359806680226482 u_vars_time_1[14]
   - 0.0229423008703687 u_vars_time_1[15]
   - 0.0126886811558233 u_vars_time_1[17]
   - 0.00278700904140223 u_vars_time_1[18] + u_vars_time_3[3]
   <= 7.584966110179809
 Const4_time_3: - 0.0168035617727886 u_vars_time_1[4]
   - 0.0107144306895757 u_vars_time_1[5]
   - 0.00592582207543151 u_vars_time_1[7]
   - 0.00130157890212156 u_vars_time_1[8]
   - 0.2463759207821404 u_vars_time_1[14]
   - 0.4458732350691355 u_vars_time_1[15]
   - 0.6146043948291333 u_vars_time_1[17]
   - 0.0783134757333082 u_vars_time_1[18]
   - 0.00107260318489199 u_vars_time_1[19] + u_vars_time_3[13]
   <= 12.83556870754241
 Const5_time_3: u_vars_time_3[0] + u_vars_time_3[10] <= 30000
 Const6_time_3: u_vars_time_3[1] + u_vars_time_3[11] <= 30000
 R25: - u_vars_time_3[8] + 0.25 u_vars_time_3[9] <= 0
 R26: - u_vars_time_3[18] + 0.25 u_vars_time_3[19] <= 0
Bounds
 u_vars_time_1[4] = 1
 u_vars_time_1[5] <= 1
 u_vars_time_1[6] <= 1
 u_vars_time_1[7] <= 1
 u_vars_time_1[8] <= 1
 u_vars_time_1[9] <= 1
 u_vars_time_1[14] = 1
 u_vars_time_1[15] <= 1
 u_vars_time_1[16] <= 1
 u_vars_time_1[17] <= 1
 u_vars_time_1[18] <= 1
 u_vars_time_1[19] <= 1
 u_vars_time_2[4] = 1
 u_vars_time_2[5] <= 1
 u_vars_time_2[6] <= 1
 u_vars_time_2[7] <= 1
 u_vars_time_2[8] <= 1
 u_vars_time_2[9] <= 1
 u_vars_time_2[14] = 1
 u_vars_time_2[15] <= 1
 u_vars_time_2[16] <= 1
 u_vars_time_2[17] <= 1
 u_vars_time_2[18] <= 1
 u_vars_time_2[19] <= 1
 u_vars_time_3[4] = 1
 u_vars_time_3[5] <= 1
 u_vars_time_3[6] <= 1
 u_vars_time_3[7] <= 1
 u_vars_time_3[8] <= 1
 u_vars_time_3[9] <= 1
 u_vars_time_3[14] = 1
 u_vars_time_3[15] <= 1
 u_vars_time_3[16] <= 1
 u_vars_time_3[17] <= 1
 u_vars_time_3[18] <= 1
 u_vars_time_3[19] <= 1
 Constant = 1
End
