#!/bin/bash

for case in b40.prescribed_carb.001 \
            b40.1850_ramp.1deg.ncbdrd.001 \
            b40.1850_ramp.1deg.ncbdrc.001 \
            b40.1850_ramp.1deg.ncbcrd.002 \
            b40.20th.1deg.bdrd.001 \
            b40.coup_carb.004 \
            b40.20th.1deg.coup.001 ; do

   echo $case
   mkdir -p /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; ls *.TOTECOSYSC.* *.NBP.* *.XSMRPOOL.* *.GPP.* *.NPP.* *.ER.* *.AR.* *.HR.*"
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; cget *.TOTECOSYSC.* *.NBP.* *.XSMRPOOL.* *.GPP.* *.NPP.* *.ER.* *.AR.* *.HR.*"

done

