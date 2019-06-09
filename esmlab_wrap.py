"""
wrappers to esmlab functionality

using these wrappers eases adapting to API changes in esmlab
"""

import esmlab

def compute_ann_mean(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_ann_mean(ds)
    return esmlab.resample(ds, freq='ann')

def compute_mon_anomaly(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_anomaly(ds)
    return esmlab.anomaly(ds, clim_freq='mon')

def compute_mon_climatology(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_climatology(ds)
    return esmlab.climatology(ds, freq='mon')
