#! /usr/bin/env python3

from utils import print_timestamp, time_set_mid, time_year_plus_frac
from xr_ds_ex import xr_ds_ex

for decode_times in [True, False]:
    print('******************************')
    print_timestamp(f'decode_times = {decode_times}')
    ds = xr_ds_ex(decode_times, nyrs=1)
    print(ds['time'])
    print('**********')
    print(ds['time'].encoding)
    print('**********')
    print('ds.time time_year_plus_frac')
    print(time_year_plus_frac(ds, 'time'))

    print('********************')
    print('calling time_set_mid on ds')

    time_set_mid(ds, 'time')
    print(ds['time'])
    print('**********')
    print(ds['time'].encoding)
    print('**********')
    print('ds.time time_year_plus_frac')
    print(time_year_plus_frac(ds, 'time'))
