#!/bin/bash

for case in b40.1850_ramp.1deg.ncbdrd.001 ; do

   echo $case

   mkdir -p /glade/scratch/klindsay/archive/$case/ocn/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/ocn/proc/tseries/monthly

   hsi "cd /CCSM/csm/$case/ocn/proc/tseries/monthly ; ls *.DIC.*"
   hsi "cd /CCSM/csm/$case/ocn/proc/tseries/monthly ; cget *.DIC.*"

done

for case in b40.prescribed_carb.001 \
            b40.20th.1deg.bdrd.001 \
            b40.rcp8_5.1deg.bdrd.001 \
            b40.coup_carb.004 \
            b40.20th.1deg.coup.001 \
            b40.rcp8_5.1deg.bprp.002 ; do

   echo $case

   mkdir -p /glade/scratch/klindsay/archive/$case/ocn/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/ocn/proc/tseries/monthly

   hsi "cd /CCSM/csm/$case/ocn/proc/tseries/monthly ; ls *.FG_CO2.* *.FvPER_DIC.* *.FvICE_DIC.*"
   hsi "cd /CCSM/csm/$case/ocn/proc/tseries/monthly ; cget *.FG_CO2.* *.FvPER_DIC.* *.FvICE_DIC.*"

done
