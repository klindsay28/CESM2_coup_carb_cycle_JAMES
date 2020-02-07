#!/bin/bash

for case in b40.prescribed_carb.001 \
            b40.1850_ramp.1deg.ncbdrd.001 \
            b40.1850_ramp.1deg.ncbdrc.001 \
            b40.1850_ramp.1deg.ncbcrd.002 \
            b40.20th.1deg.bdrd.001 \
            b40.rcp8_5.1deg.bdrd.001 \
            b40.coup_carb.004 \
            b40.20th.1deg.coup.001 \
            b40.rcp8_5.1deg.bprp.002 ; do

   echo $case
   mkdir -p /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   # physical state
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; ls *.TSA.* *.RAIN.* *.TSOI.* *.H2OSOI.* *.SOILLIQ.*"
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; cget *.TSA.* *.RAIN.* *.TSOI.* *.H2OSOI.* *.SOILLIQ.*"
   # carbon pools
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; ls *.TOTECOSYSC.* *.TOTVEGC.* *.CWDC.* *.TOTLITC.* *.TOTSOMC.* *.TOTPRODC.* *.XSMRPOOL.*"
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; cget *.TOTECOSYSC.* *.TOTVEGC.* *.CWDC.* *.TOTLITC.* *.TOTSOMC.* *.TOTPRODC.* *.XSMRPOOL.*"
   # carbon fluxes
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; ls *.NBP.* *.GPP.* *.NPP.* *.ER.* *.AR.* *.HR.* *.COL_FIRE_CLOSS.*"
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; cget *.NBP.* *.GPP.* *.NPP.* *.ER.* *.AR.* *.HR.* *.COL_FIRE_CLOSS.*"

done

