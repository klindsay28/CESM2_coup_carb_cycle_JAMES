#! /usr/bin/env python3

from esmlab_wrap import compute_ann_mean
from utils import print_timestamp, time_year_plus_frac
from xr_ds_ex import xr_ds_ex

for decode_times in [True]:
    print('******************************')
    print_timestamp(f'decode_times = {decode_times}')
    ds = xr_ds_ex(decode_times)
    print(ds)

    print('********************')
    ds_ann = compute_ann_mean(ds)
    print(ds_ann)
    print('**********')
    print(ds_ann.attrs['history'])

    print('********************')
    ds.attrs['history'] = 'test'
    ds_ann = compute_ann_mean(ds)
    print(ds_ann)
    print('**********')
    print(ds_ann.attrs['history'])
