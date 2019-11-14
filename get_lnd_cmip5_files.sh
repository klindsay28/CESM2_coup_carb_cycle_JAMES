#!/bin/bash

for case in b40.prescribed_carb.001 \
            b40.coup_carb.004 ; do

   echo $case
   mkdir -p /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/lnd/proc/tseries/monthly
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; ls *.TOTECOSYSC.* *.NBP.* *.XSMRPOOL.*"
   hsi "cd /CCSM/csm/$case/lnd/proc/tseries/monthly ; cget *.TOTECOSYSC.* *.NBP.* *.XSMRPOOL.*"

done

