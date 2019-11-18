"""
wrappers to esmlab functionality

using these wrappers eases adapting to API changes in esmlab
"""

import esmlab

def compute_ann_mean(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_ann_mean(ds)
    ds_ann = esmlab.resample(ds, freq='ann')

    tb_name = ds_ann.time.bounds
    if ds_ann[tb_name].dims[0] != 'time':
        ds_ann[tb_name] = ds_ann[tb_name].transpose()

    for key in ['unlimited_dims']:
        if key in ds.encoding:
            ds_ann.encoding[key] = ds.encoding[key]

    return ds_ann

def compute_mon_anomaly(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_anomaly(ds)
    return esmlab.anomaly(ds, clim_freq='mon')
