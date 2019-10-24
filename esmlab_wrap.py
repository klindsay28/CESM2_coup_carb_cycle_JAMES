"""
wrappers to esmlab functionality

using these wrappers eases adapting to API changes in esmlab
"""

import esmlab

def compute_ann_mean(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_ann_mean(ds)
    ds_ann = esmlab.resample(ds, freq='ann')
    ds_ann['time'] = ds_ann['time'] # ensures indices in ds_ann for time are correct
    return ds_ann

def compute_mon_anomaly(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_anomaly(ds)
    return esmlab.anomaly(ds, clim_freq='mon')

def compute_mon_climatology(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_climatology(ds)
    return esmlab.climatology(ds, freq='mon')
