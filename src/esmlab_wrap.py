"""
wrappers to esmlab functionality

using these wrappers eases adapting to API changes in esmlab
"""

from datetime import datetime, timezone
import warnings

from esmlab import resample, anomaly

from src.utils import time_set_mid

def compute_ann_mean(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_ann_mean(ds)
    # ignore certain warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module='.*nanops')
        ds_out = resample(ds, freq='ann')

    # esmlab appears to corrupt xarray indexes wrt time
    # the following seems to reset them
    ds_out['time'] = ds_out['time']

    # ensure time dim is first on time.bounds variable
    tb_name = ds_out.time.bounds
    if ds_out[tb_name].dims[0] != 'time':
        ds_out[tb_name] = ds_out[tb_name].transpose()

    # reset time to midpoint
    ds_out = time_set_mid(ds_out, 'time')

    # propagate particular encoding values
    for key in ['unlimited_dims']:
        if key in ds.encoding:
            ds_out.encoding[key] = ds.encoding[key]

    # copy file attributes
    for key in ds.attrs:
        if key != 'history':
            ds_out.attrs[key] = ds.attrs[key]

    # append to history file attribute if it already exists, otherwise set it
    key = 'history'
    datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    msg = f'{datestamp}: created by esmlab.resample, with modifications from esmlab_wrap'
    if key in ds.attrs:
        ds_out.attrs[key] = '\n'.join([msg, ds.attrs[key]])
    else:
        ds_out.attrs[key] = msg

    return ds_out

def compute_mon_anomaly(ds):
    """esmlab wrapper"""
#     return esmlab.climatology.compute_mon_anomaly(ds)
    ds_out = anomaly(ds, clim_freq='mon')

    # propagate particular encoding values
    for key in ['unlimited_dims']:
        if key in ds.encoding:
            ds_out.encoding[key] = ds.encoding[key]

    # copy file attributes, prepending history message
    for key in ds.attrs:
        if key == 'history':
            datestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            msg = f'{datestamp}: created by esmlab.anomaly, with modifications from esmlab_wrap'
            ds_out.attrs[key] = '\n'.join([msg, ds.attrs[key]])
        else:
            ds_out.attrs[key] = ds.attrs[key]

    return ds_out
