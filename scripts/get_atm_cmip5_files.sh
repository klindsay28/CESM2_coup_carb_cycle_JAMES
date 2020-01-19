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
   mkdir -p /glade/scratch/klindsay/archive/$case/atm/proc/tseries/monthly
   cd /glade/scratch/klindsay/archive/$case/atm/proc/tseries/monthly
   hsi "cd /CCSM/csm/$case/atm/proc/tseries/monthly ; ls *CO2* *.PS.* *SFCO2* *TMCO2* *TS.*"
   hsi "cd /CCSM/csm/$case/atm/proc/tseries/monthly ; cget *CO2* *.PS.* *SFCO2* *TMCO2* *TS.*"

done

