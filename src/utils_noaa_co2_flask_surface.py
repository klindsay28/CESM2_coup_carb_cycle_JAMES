"""utilities for NOAA co2_flask_surface files"""

import cftime
import numpy as np
import xarray as xr


def co2_flask_surface_read_file(
    fname_co2_flask_surface, encode_time=False, calendar="noleap", year_ref=1
):
    """
    Return a Dataset of values from a NOAA co2_flask_surface file.
    If encode_time==False, time values are cftime objects.
    Otherwise they are floating point numbers in units of "days since {year_ref}-01-01",
    with calendar=calendar.
    """
    with open(fname_co2_flask_surface) as fobj:
        # get number of header lines from 1st line
        number_of_header_lines = int(fobj.readline().split()[-1])

        # read header, storing contents in a dict for Dataset metadata
        header = {}
        for line_no in range(1, number_of_header_lines):
            # read line, split into att_name and att_value, ignoring 1st 2 characters
            [att_name, att_value] = fobj.readline()[2:].split(":", 1)
            # ignore leading space from att_value, strip trailing newline
            att_value = att_value[1:-1]
            if att_name in header:
                header[att_name] = "\n".join([header[att_name], att_value])
            else:
                header[att_name] = att_value

        # read data
        data_fields_names = header["data_fields"].split()
        data = {data_field_name: [] for data_field_name in data_fields_names}
        for line in iter(fobj.readline, ""):
            line_split = line.split()
            for field_ind, data_field_name in enumerate(data_fields_names):
                data[data_field_name].append(line_split[field_ind])

    ds = gen_time_vars(data)

    # add data fields to Dataset
    for data_field_name in data_fields_names:
        try:
            values = [int(value) for value in data[data_field_name]]
        except ValueError:
            try:
                values = [float(value) for value in data[data_field_name]]
            except ValueError:
                values = data[data_field_name]
        da = xr.DataArray(values, dims="time", coords={"time": ds["time"]})
        if data_field_name in ["value", "analysis_value"]:
            name = "CO2"
            da.attrs["units"] = "ppmv"
        else:
            name = data_field_name
        ds[name] = da

    # add global metadata to Dataset atributes
    for key, value in header.items():
        ds.attrs[key] = value
    ds.attrs["input_file_list"] = fname_co2_flask_surface

    return ds


def gen_time_vars(data, encode_time=False, calendar="noleap", year_ref=1):
    """
    Return Dataset with monthly time and time_bounds DataArrays spanning year_beg to year_end
    Generated values include year_end.
    If encode_time==False, time values are cftime objects.
    Otherwise they are floating point numbers in units of "days since {year_ref}-01-01",
    with calendar=calendar.
    """

    if "sample_year" in data:
        raise NotImplementedError(f"sample data not implemented")

    # select which cftime class will be used, based on calendar
    cftime_class = None
    if calendar in ["gregorian", "standard"]:
        cftime_class = cftime.DatetimeGregorian
    if calendar in ["proleptic_gregorian"]:
        cftime_class = cftime.DatetimeProlepticGregorian
    if calendar in ["noleap", "365_day"]:
        cftime_class = cftime.DatetimeNoLeap
    if cftime_class is None:
        raise NotImplementedError(f"calendar={calendar} not implemented")

    value_cnt = len(data["value"])
    time_edges_values = np.empty(value_cnt + 1, dtype=np.dtype("O"))
    for ind in range(value_cnt):
        year = int(data["year"][ind])
        month = int(data["month"][ind])
        time_edges_values[ind] = cftime_class(year, month, 1)
    ind += 1
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    time_edges_values[ind] = cftime_class(year, month, 1)

    time_bounds_values = np.stack(
        (time_edges_values[:-1], time_edges_values[1:]), axis=1
    )
    time_values = time_bounds_values[:, 0] + 0.5 * (
        time_bounds_values[:, 1] - time_bounds_values[:, 0]
    )

    time_units = f"days since {year_ref:04d}-01-01"

    # encode time, if requested
    if encode_time == True:
        time_bounds_values = cftime.date2num(
            time_bounds_values, units=time_units, calendar=calendar
        )
        time_values = cftime.date2num(time_values, units=time_units, calendar=calendar)

    time_var = xr.DataArray(
        time_values,
        name="time",
        dims="time",
        coords={"time": time_values},
        attrs={"bounds": "time_bounds"},
    )
    time_bounds_var = xr.DataArray(
        time_bounds_values,
        name="time_bounds",
        dims=("time", "d2"),
        coords={"time": time_var},
    )

    time_var.encoding["_FillValue"] = None
    time_var.encoding["dtype"] = "float64"
    time_bounds_var.encoding["_FillValue"] = None
    time_bounds_var.encoding["dtype"] = "float64"

    # store metadata appropriately
    if encode_time == True:
        time_var.attrs["units"] = time_units
        time_var.attrs["calendar"] = calendar
    else:
        time_var.encoding["units"] = time_units
        time_var.encoding["calendar"] = calendar

    return xr.Dataset({"time": time_var, "time_bounds": time_bounds_var})
