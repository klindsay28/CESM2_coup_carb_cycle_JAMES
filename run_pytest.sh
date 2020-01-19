#!/bin/bash -i
conda activate CESM2_coup_carb_cycle_JAMES_tst
python -m pytest tests/test_utils.py tests/test_utils_units.py tests/test_esmlab_wrap.py
# python -m pytest tests/test_tseries_mod.py
