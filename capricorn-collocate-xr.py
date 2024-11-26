#!/usr/bin/env python3
# coding: utf-8
# capricorn-collocate-xy.py
# code to collocate model output with CAP ship using xarray
# (originally used iris but find this cleaner)

# Inputs:
# suite:     UM suite code
# config:    string used to define model configuration for suite (as set
#            in rose namelists and used by me in filenames)
# stashcode: UM stashcode of variable to collocate

# RP July 2024
# ======================================================================
import numpy as np
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import sys
sys.path.append("/home/users/eersp/cap_notebooks/")
from capricorn_functions import get_cap_coords
import argparse
import os

# ----------------------------------------------------------------------
# functions
def get_closest_time(time, target_times):
    # Function to take a given time (dt object) and use np.argmin
    # to find the closest in time of an array of other timesteps,
    # target_times (array of dt objects). 
    # Returns the index of target_times with the resulting time.
    # Note haven't tested full range of behaviour
    # eg if two timesteps are equally close, or if time outside range of
    # target_time.
    dT = [abs((target_time-time).total_seconds())
          for target_time in target_times]
    ind = np.argmin(dT)
    return ind

start = dt.datetime.now()
# define suite from command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="u-cr700")
parser.add_argument("--config", default="")
parser.add_argument("--stashcode", default="m01s00i004")
args = parser.parse_args()
suite = args.suite
config = args.config
stashcode = args.stashcode
print(f'\n\n[STASH] processing {stashcode} from {suite}{config}')

# parameters for datetime conversions
unix_epoch_npy = np.datetime64(0, 's')
unix_epoch_dt = dt.datetime(1970,1,1)
one_second = np.timedelta64(1, 's')
ref_time_dt = dt.datetime(2018,1,31)
ref_time_s = (ref_time_dt-unix_epoch_dt).total_seconds()

# ----------------------------------------------------------------------
# LOAD DATA
# load CAP nav info into 1d arrays of time, lat, lon
cap_times, cap_lons, cap_lats = get_cap_coords()

# load model data
rad_stash = [
    'm01s01i201','m01s01i210','m01s01i211','m01s01i235',
    'm01s02i201','m01s02i207','m01s02i208',
]
print(f'[INFO] loading model data processed using cap_cdo_output.sh')
model_output_dir = f'/gws/nopw/j04/asci/rprice/ukca_casim_output/{suite}/'
filename = f'postproc/STASH_{stashcode}.nc'
model_output = xr.open_dataset(model_output_dir+filename)
if stashcode in rad_stash:
    time_npy = model_output['T1HR_rad_diag'].values
else:
    time_npy = model_output['T1HR'].values          # npy datetime objects
model_times = (time_npy-unix_epoch_npy)/one_second  # seconds since epoch
model_times = np.array([unix_epoch_dt+dt.timedelta(seconds=T)
                        for T in model_times])
# for each element of model_times, get index of cap_times at that time
i_Tmodel_to_Tship = np.array([get_closest_time(T, cap_times)
                              for T in model_times])
# take subset of CAPRICORN coords that are at model timesteps
cap_lons = cap_lons[i_Tmodel_to_Tship]
cap_lats = cap_lats[i_Tmodel_to_Tship]
# get CAP coords on model rotated pole projection
rot_pole_coords = model_output.rotated_latitude_longitude
rot_pole_lon = rot_pole_coords.grid_north_pole_longitude
rot_pole_lat = rot_pole_coords.grid_north_pole_latitude
rot_pole = ccrs.RotatedPole(pole_latitude=rot_pole_lat,
                            pole_longitude=rot_pole_lon)
cap_rlons = rot_pole.transform_points(src_crs=ccrs.PlateCarree(),
                                      x=cap_lons, y=cap_lats)[:,0]
cap_rlons[cap_rlons<0] += 360
cap_rlons[cap_rlons>360] -= 360
cap_rlats = rot_pole.transform_points(src_crs=ccrs.PlateCarree(),
                                      x=cap_lons, y=cap_lats)[:,1]

# ----------------------------------------------------------------------
# PERFORM COLLOCATION
print('[INFO] starting collocation')
# nearest neighbour only
nearest_neighbour = False
if nearest_neighbour:
    model_output_cap = []
    for i in np.arange(len(cap_rlons)):
        model_output_t = model_output.isel(T1HR=i)
        model_output_lat = model_output_t.sel(grid_latitude_t=cap_rlats[i],
                                              method='nearest')
        model_output_lon = model_output_lat.sel(grid_longitude_t=cap_rlons[i],
                                                method='nearest')
        model_output_cap.append(model_output_lon)
    model_output_cap = xr.concat(model_output_cap, dim='T1HR')

# 9x9 nearest model points
dx = np.diff(model_output.grid_longitude_t.values)[0]
dy = np.diff(model_output.grid_latitude_t.values)[0]
model_output_cap_mean = []
mean_grid_lon = []
mean_grid_lat = []
mean_lon = []
mean_lat = []
model_output_grid = []
for i in np.arange(len(cap_rlons)):
    if stashcode in rad_stash:
        model_output_t = model_output.isel(T1HR_rad_diag=i)
    else:
        model_output_t = model_output.isel(T1HR=i)
    xx = slice(cap_rlons[i]-1.5*dx,cap_rlons[i]+1.5*dx)
    yy = slice(cap_rlats[i]-1.5*dy,cap_rlats[i]+1.5*dy)
    model_output_lat = model_output_t.sel(grid_latitude_t=yy)
    model_output_lon = model_output_lat.sel(grid_longitude_t=xx)
    var = model_output_lon[f'STASH_{stashcode}']
    # reshape lat/lon dimensions into 1d, length 9 grid
    if var.shape[-2:] == (3, 3):
        new_shape = var.shape[:-2] + (9,)
    else:
        raise ValueError(f"Unexpected data shape for {stashcode}: {var.shape}")
    new_coords = {}
    for dim in var.dims[:-2]:
        new_coords[dim] = var.coords[dim]
    if stashcode in rad_stash:
        new_coords['T1HR_rad_diag'] = var['T1HR_rad_diag']
    else:
        new_coords['T1HR'] = var['T1HR']
    new_coords['longitude_t'] = ('grid_number', var['longitude_t'].values.reshape(9))
    new_coords['latitude_t'] = ('grid_number', var['longitude_t'].values.reshape(9))
    yy, xx = np.meshgrid(
        var['grid_latitude_t'].values,
        var['grid_longitude_t'].values,
        indexing='ij')
    new_coords['grid_latitude_t'] = ('grid_number', yy.flatten())
    new_coords['grid_longitude_t'] = ('grid_number', xx.flatten())
    new_coords['grid_number'] = np.arange(9)
    # save 9x9 grid
    var_grid = xr.DataArray(
        data=var.data.reshape(new_shape),
        dims=var.dims[:-2] + ('grid_number',),
        coords=new_coords,
        name=f'STASH_{stashcode}'
    )
    model_output_grid.append(var_grid)
    # save spatial mean and preserve mean of lat and lon
    mean_var = var.mean(dim=['grid_latitude_t','grid_longitude_t'])
    mean_x = var.coords['grid_longitude_t'].mean().item()
    mean_y = var.coords['grid_latitude_t'].mean().item()
    mean_grid_lon.append(mean_x)
    mean_grid_lat.append(mean_y)
    mean_x = var.coords['longitude_t'].mean().item()
    mean_y = var.coords['latitude_t'].mean().item()
    mean_lon.append(mean_x)
    mean_lat.append(mean_y)
    model_output_cap_mean.append(mean_var)
model_output_cap_mean = xr.concat(model_output_cap_mean, dim='T1HR')
model_output_cap_mean.attrs['mean_grid_longitude'] = mean_grid_lon
model_output_cap_mean.attrs['mean_grid_latitude'] = mean_grid_lat
model_output_cap_mean.attrs['mean_longitude'] = mean_lon
model_output_cap_mean.attrs['mean_latitude'] = mean_lat
model_output_cap_grid = xr.concat(model_output_grid, dim='T1HR')

# ----------------------------------------------------------------------
# SAVE OUTPUT
save_dir = f'data/model/'
try:
    os.mkdir(save_dir)
except FileExistsError:
    pass
# save spatial mean
save_fn = f'{suite}_{stashcode}_CAP.nc'
print(f'[INFO] saving to {save_dir}{save_fn}')
model_output_cap_mean.to_netcdf(save_dir+save_fn)
# save 9x9 gridded output
save_fn = f'{suite}_{stashcode}_grid_CAP.nc'
print(f'[INFO] saving to {save_dir}{save_fn}')
model_output_cap_grid.to_netcdf(save_dir+save_fn)

# ======================================================================
end = dt.datetime.now()
print(f'\n\n[END] finished in {end-start}')
