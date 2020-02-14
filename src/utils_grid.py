"""grid related utility functions"""

from collections import OrderedDict

import numpy as np
import xarray as xr

from src.utils import print_timestamp, dim_cnt_check
from src.CIME_shr_const import CIME_shr_const


def get_weight(ds, component, reduce_dims):
    """construct averaging/integrating weight appropriate for component and reduce_dims"""
    if component == "lnd":
        return get_area(ds, component)
    if component == "ice":
        return get_area(ds, component)
    if component == "atm":
        if "lev" in reduce_dims:
            return get_volume(ds, component)
        return get_area(ds, component)
    if component == "ocn":
        if "z_t" in reduce_dims or "z_t_150m" in reduce_dims:
            return get_volume(ds, component)
        return get_area(ds, component)
    msg = f"unknown component={component}"
    raise ValueError(msg)


def get_area(ds, component):
    """return area DataArray appropriate for component"""
    if component == "ocn":
        dim_cnt_check(ds, "TAREA", 2)
        return ds["TAREA"]
    if component == "ice":
        dim_cnt_check(ds, "tarea", 2)
        return ds["tarea"]
    if component == "lnd":
        dim_cnt_check(ds, "landfrac", 2)
        dim_cnt_check(ds, "area", 2)
        da_ret = ds["landfrac"] * ds["area"]
        da_ret.name = "area"
        da_ret.attrs["units"] = ds["area"].attrs["units"]
        return da_ret
    if component == "atm":
        dim_cnt_check(ds, "gw", 1)
        area_earth = (
            4.0 * CIME_shr_const("pi") * CIME_shr_const("rearth") ** 2
        )  # area of earth in CIME [m2]

        # normalize area so that sum over "lat", "lon" yields area_earth
        area = ds["gw"] + 0.0 * ds["lon"]  # add "lon" dimension
        area = (area_earth / area.sum(dim=("lat", "lon"))) * area
        area.attrs["units"] = "m2"
        return area
    msg = f"unknown component={component}"
    raise ValueError(msg)


def get_volume(ds, component):
    """return volume DataArray appropriate for component"""
    if component == "ocn":
        dim_cnt_check(ds, "dz", 1)
        dim_cnt_check(ds, "TAREA", 2)
        volume = ds["dz"] * ds["TAREA"]
        volume.attrs["units"] = "cm3"
        return volume

    msg = f"component={component} not implemented"
    raise NotImplementedError(msg)


def get_rmask(ds, component):
    """return region mask appropriate for component"""
    rmask_od = OrderedDict()
    if component == "ocn":
        dim_cnt_check(ds, "KMT", 2)
        lateral_dims = ds["KMT"].dims
        # treat missing-values, which arise from land-block elimination as 0
        KMT = ds["KMT"].fillna(0).load()
        REGION_MASK = ds["REGION_MASK"].fillna(0).load()
        TLAT = ds["TLAT"].load()
        TLONG = ds["TLONG"].load()

        # include KMT > 0 condition in region definitions
        # TLAT and TLONG may be _FillValue, because of land-block elimination

        rmask_od["Global"] = xr.where(KMT > 0, 1.0, 0.0)
        rmask_od["SouOce (90S-30S)"] = xr.where((KMT > 0) & (TLAT < -30.0), 1.0, 0.0)
        rmask_od["SH_high_lat (90S-44S)"] = xr.where(
            (KMT > 0) & (TLAT < -44.0), 1.0, 0.0
        )
        rmask_od["SH_high_lat_ATL (90S-44S)"] = xr.where(
            (KMT > 0) & (TLAT < -44.0) & ((TLONG < 20.0) | (TLONG >= 291.0)), 1.0, 0.0
        )
        rmask_od["SH_high_lat_IND (90S-44S)"] = xr.where(
            (KMT > 0) & (TLAT < -44.0) & (TLONG >= 20.0) & (TLONG < 147.0), 1.0, 0.0
        )
        rmask_od["SH_high_lat_PAC (90S-44S)"] = xr.where(
            (KMT > 0) & (TLAT < -44.0) & (TLONG >= 147.0) & (TLONG < 291.0), 1.0, 0.0
        )
        rmask_od["SH_mid_lat_ATL (44S-18S)"] = xr.where(
            (KMT > 0)
            & (TLAT >= -44.0)
            & (TLAT < -18.0)
            & ((TLONG < 20.0) | (TLONG >= 291.0)),
            1.0,
            0.0,
        )
        rmask_od["SH_mid_lat_IND (44S-18S)"] = xr.where(
            (KMT > 0)
            & (TLAT >= -44.0)
            & (TLAT < -18.0)
            & (TLONG >= 20.0)
            & (TLONG < 147.0),
            1.0,
            0.0,
        )
        rmask_od["SH_mid_lat_PAC (44S-18S)"] = xr.where(
            (KMT > 0)
            & (TLAT >= -44.0)
            & (TLAT < -18.0)
            & (TLONG >= 147.0)
            & (TLONG < 291.0),
            1.0,
            0.0,
        )
        rmask_od["low_lat_ATL (18S-18N)"] = xr.where(
            (KMT > 0) & (TLAT >= -18.0) & (TLAT < 18.0) & (REGION_MASK == 6), 1.0, 0.0
        )
        rmask_od["low_lat_IND (18S-27N)"] = xr.where(
            (KMT > 0) & (TLAT >= -18.0) & (TLAT < 18.0) & (REGION_MASK == 3), 1.0, 0.0
        )
        rmask_od["low_lat_PAC (18S-18N)"] = xr.where(
            (KMT > 0) & (TLAT >= -18.0) & (TLAT < 18.0) & (REGION_MASK == 2), 1.0, 0.0
        )
        rmask_od["NH_mid_lat_ATL (18N-45N)"] = xr.where(
            (KMT > 0)
            & (TLAT >= 18.0)
            & (TLAT < 45.0)
            & (REGION_MASK >= 6)
            & (REGION_MASK < 9),
            1.0,
            0.0,
        )
        rmask_od["NH_mid_lat_PAC (18N-45N)"] = xr.where(
            (KMT > 0) & (TLAT >= 18.0) & (TLAT < 45.0) & (REGION_MASK == 2), 1.0, 0.0
        )
        rmask_od["NH_subpolar_ATL (45N-90N)"] = xr.where(
            (KMT > 0) & (TLAT >= 45.0) & (REGION_MASK >= 6) & (REGION_MASK < 9),
            1.0,
            0.0,
        )
        rmask_od["NH_high_lat_PAC (45N-90N)"] = xr.where(
            (KMT > 0) & (TLAT >= 45.0) & (REGION_MASK == 2), 1.0, 0.0
        )
        rmask_od["GIN"] = xr.where((KMT > 0) & (REGION_MASK == 9), 1.0, 0.0)
        rmask_od["ARC"] = xr.where((KMT > 0) & (REGION_MASK == 10), 1.0, 0.0)
    if component == "ice":
        dim_cnt_check(ds, "tmask", 2)
        dim_cnt_check(ds, "TLAT", 2)
        lateral_dims = ds["tmask"].dims
        tmask = ds["tmask"].load()
        TLAT = ds["TLAT"].load()
        rmask_od["NH"] = xr.where((tmask == 1) & (TLAT >= 0.0), 1.0, 0.0)
        rmask_od["SH"] = xr.where((tmask == 1) & (TLAT < 0.0), 1.0, 0.0)
    if component == "lnd":
        dim_cnt_check(ds, "landfrac", 2)
        lateral_dims = ds["landfrac"].dims
        lat = ds["lat"].load()
        lon = ds["lon"].load()
        rmask_od["Global"] = xr.where(ds["landfrac"] > 0, 1.0, 0.0)
        rmask_od["CentralAfrica"] = xr.where(
            (ds["landfrac"] > 0)
            & (lat >= -10.0)
            & (lat < 10.0)
            & (lon >= 0.0)
            & (lon < 55.0),
            1.0,
            0.0,
        )
        rmask_od["MaritimeContinent"] = xr.where(
            (ds["landfrac"] > 0)
            & (lat >= -11.0)
            & (lat < 8.0)
            & (lon >= 90.0)
            & (lon < 160.0),
            1.0,
            0.0,
        )
        rmask_od["Australia"] = xr.where(
            (ds["landfrac"] > 0)
            & (lat >= -45.0)
            & (lat < -11.0)
            & (lon >= 110.0)
            & (lon < 160.0),
            1.0,
            0.0,
        )
        rmask_od["TropSAmer"] = xr.where(
            (ds["landfrac"] > 0)
            & (lat >= -15.0)
            & (lat < 15.0)
            & (lon >= 278.0)
            & (lon < 330.0),
            1.0,
            0.0,
        )
        rmask_od["SSAmer"] = xr.where(
            (ds["landfrac"] > 0)
            & (lat >= -60.0)
            & (lat < -15.0)
            & (lon >= 278.0)
            & (lon < 330.0),
            1.0,
            0.0,
        )
    if component == "atm":
        dim_cnt_check(ds, "gw", 1)
        lateral_dims = ("lat", "lon")
        lat = ds["lat"].load()
        lon = ds["lon"].load()
        rmask_od["Global"] = xr.where((lat > -100.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od["SH"] = xr.where((lat < 0.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od["SH_Trop"] = xr.where(
            (lat > -30) & (lat < 0.0) & (lon > -400.0), 1.0, 0.0
        )
        rmask_od["NH"] = xr.where((lat > 0.0) & (lon > -400.0), 1.0, 0.0)
        rmask_od["NH_Trop"] = xr.where(
            (lat > 0.0) & (lat < 30.0) & (lon > -400.0), 1.0, 0.0
        )
        rmask_od["nino34"] = xr.where(
            (lat > -5.0) & (lat < 5.0) & (lon > 190) & (lon < 240), 1.0, 0.0
        )
    if len(rmask_od) == 0:
        msg = f"unknown component={component}"
        raise ValueError(msg)

    print_timestamp("rmask_od created")

    rmask = xr.DataArray(
        np.zeros((len(rmask_od), ds.dims[lateral_dims[0]], ds.dims[lateral_dims[1]])),
        dims=("region", lateral_dims[0], lateral_dims[1]),
        coords={"region": list(rmask_od.keys())},
    )
    rmask.region.encoding["dtype"] = "S1"

    # add coordinates if appropriate
    if component == "atm" or component == "lnd":
        rmask.coords["lat"] = ds["lat"]
        rmask.coords["lon"] = ds["lon"]

    for i, rmask_field in enumerate(rmask_od.values()):
        rmask.values[i, :, :] = rmask_field

    return rmask
