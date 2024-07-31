#!/usr/bin/env python3
# coding: utf-8
# capricorn_functions.py 
# Functions used in analysis and plotting of UM CAPRICORN-2 case studies

# =====================================================================
import xarray as xr
import pandas as pn
import datetime as dt
import numpy as np
from glob import glob
import capricorn_constants as const

# ---------------------------------------------------------------------

def get_cap_coords(filename='./data/CAPRICORN/in2018_v01uwy1min.csv'):
    # Load CAP2 lat/lon coords (with timesteps) from file
    nav_df = pn.read_csv(filename, parse_dates=True, low_memory=False)
    # save times as dt objects (easier to manipulate)
    cap_nav_times = []
    fmt_date = '%d-%b-%Y %H:%M:%S'
    for i in np.arange(nav_df["date"].shape[0]):
        time_str = f'{nav_df["date"][i]} {nav_df["time"][i]}'
        cap_nav_times.append(dt.datetime.strptime(time_str, fmt_date))
    cap_nav_times = np.array(cap_nav_times)
    # save lat and lons into arrays and deal with missing data
    fill_val = -99999
    cap_lons = np.empty((nav_df["longitude(degree_east)"].shape[0]))
    cap_lats = np.empty((nav_df["latitude(degree_north)"].shape[0]))
    for x in np.arange(cap_lons.shape[0]):
        try:
            cap_lons[x] = float(nav_df["longitude(degree_east)"][x])
        except ValueError:
            cap_lons[x] = fill_val
    for y in np.arange(cap_lats.shape[0]):
        try:
            cap_lats[y] = float(nav_df["latitude(degree_north)"][y])
        except ValueError:
            cap_lats[y] = fill_val
    cap_lons = np.ma.masked_equal(cap_lons, fill_val)
    cap_lats = np.ma.masked_equal(cap_lats, fill_val)
    return cap_nav_times, cap_lons, cap_lats

def check_file_exists(suite, stash, verbose=False):
    '''
    For given simulation (suite) and variable (stash), check if
    collocated model output has been saved. If verbose, print warning if
    file is not there.
    '''
    fn = glob(f'data/model/{suite}_{stash}_CAP.nc')
    if len(fn)==0:
        if verbose:
            print(f'[WARNING] no file for {suite} {stash}')
        flag = False

    else:
        flag = True
    return flag

def temperature(theta, pressure):
    # function to calculate T [K] from potential T [K] and P [Pa]
    temperature = theta*(pressure/const.p0)**(const.Rd_cp)
    return temperature

def air_density(temperature, pressure):
    # function to calculate air density [kg m-3] from T [K] and P [Pa]
    air_density = (pressure/(temperature*const.Rd))
    return air_density

def calculate_air_density(suite, grid=False):
    '''
    Calculate air density from collocated theta and pressure and save (in place)
    '''
    if grid:
        theta_stash = "m01s00i004_grid"
        pressure_stash = "m01s00i408_grid"
    else:
        theta_stash = "m01s00i004"
        pressure_stash = "m01s00i408"
    theta_flag = check_file_exists(suite, theta_stash)
    if not theta_flag:
        print(f'[air_density] [WARNING] no pot. temp. for {suite}')
        print(f'[air_density] [WARNING] exiting function')
        return
    pressure_flag = check_file_exists(suite, pressure_stash)
    if not theta_flag:
        print(f'[air_density] [WARNING] no pressure for {suite}')
        print(f'[air_density] [WARNING] exiting function')
        return
    theta = xr.open_dataset(f'data/model/{suite}_{theta_stash}_CAP.nc')
    pressure = xr.open_dataset(f'data/model/{suite}_{pressure_stash}_CAP.nc')
    temp = temperature(theta['STASH_m01s00i004'], pressure['STASH_m01s00i408'])
    rho = air_density(temp, pressure['STASH_m01s00i408'])
    rho = rho.rename('air_density')
    rho.attrs['units'] = 'kg m-3'
    if grid:
        save_fn = f'data/model/{suite}_air_density_grid_CAP.nc'
    else:
        save_fn = f'data/model/{suite}_air_density_CAP.nc'
    rho.to_netcdf(save_fn)
    return

def load_air_density(suite, grid=False):
    '''
    Check if air density has already been saved. Load if saved,
    calculate and load if not.
    '''
    if grid:
        var = "air_density_grid"
    else:
        var = "air_density"
    flag = check_file_exists(suite, var)
    if not flag:
        calculate_air_density(suite, grid)
    air_density = xr.open_dataset(f'data/model/{suite}_{var}_CAP.nc')
    return air_density

def get_hist_bins(x):
    '''
    Calculate linearly spaced, fixed width bins for distrbution of array x
    '''
    # check for nans
    x = np.ma.masked_invalid(x).compressed()
    ndata = len(x)
    min = np.ma.amin(x)
    max = np.ma.amax(x)
    mean = np.ma.mean(x)
    median = np.ma.median(x)
    nbins = int(np.ceil(np.sqrt(ndata)))
    bins = np.linspace(min, max, nbins)
    return bins, [min, max, mean, median]
    