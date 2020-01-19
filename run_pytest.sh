#!/bin/bash -i
conda activate CESM2_coup_carb_cycle_JAMES_tst
python -m pytest test_utils.py test_utils_units.py test_esmlab_wrap.py
# python -m pytest test_tseries_mod.py
