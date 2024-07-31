#!/usr/bin/env python3
# coding: utf-8
# compute-domain-lwp.py
# Script to calculate and save 3D LWP over coastal domain from 3D
# liquid mass and cloud fraction fields, to compare with AMSR2
# =====================================================================
import numpy as np
import numpy.ma as ma
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pn
import sys
sys.path.append("/home/users/eersp/cap_notebooks/")
import capricorn_constants as const
import capricorn_functions as cfn
from simulations import simulations

# ---------------------------------------------------------------------
start = dt.datetime.now()
t0 = dt.datetime.now()
isims = [0,-1]
simulations = simulations[isims]
resns = 0.018
n_sims = len(simulations)
model_output_dir = '/gws/nopw/j04/asci/rprice/ukca_casim_output/'
liq_stash = 'm01s00i254'
theta_stash = 'm01s00i004'
pressure_stash = 'm01s00i408'
liq_fn = f'postproc/STASH_{liq_stash}.nc'
theta_fn = f'postproc/STASH_{theta_stash}.nc'
pressure_fn = f'postproc/STASH_{pressure_stash}.nc'
for s,sim in enumerate(simulations):
    # load output fields
    sim_output_dir = f'{model_output_dir}{sim.code}/'
    print(f'\n\n[INFO] {sim.code}: loading cloud liquid content', flush=True)
    liq_mass = xr.open_dataset(sim_output_dir+liq_fn)
    # faff with metadata
    grid_lat_bounds = liq_mass.grid_latitude_t_bnds
    grid_lon_bounds = liq_mass.grid_longitude_t_bnds
    rot_lat_lon = liq_mass.rotated_latitude_longitude
    altitude = liq_mass.TH_1_60_eta_theta.values*const.ztop
    liq_mass = liq_mass[f'STASH_{liq_stash}']
    liq_mass.grid_latitude_t.attrs.pop('bounds')
    liq_mass.grid_longitude_t.attrs.pop('bounds')
    print(f'\n\n[INFO] {sim.code}: loading theta', flush=True)
    theta = xr.open_dataset(sim_output_dir+theta_fn)[f'STASH_{theta_stash}']
    print(f'\n\n[INFO] {sim.code}: loading pressure', flush=True)
    pressure = xr.open_dataset(sim_output_dir+pressure_fn)[f'STASH_{pressure_stash}']
    print(F'\n\n[INFO] {sim.code} calculating temperature and air density')
    air_density = cfn.air_density(cfn.temperature(theta, pressure), pressure)
    print(F'\n\n[INFO] {sim.code} calculating LWC and LWP')
    lwc = liq_mass*air_density  # kg m-3
    # get altitude of model levels and calculate dz
    dz = np.zeros(altitude.shape)
    dz[0] = altitude[0]
    dz[1:] = np.diff(altitude)
    dz = np.expand_dims(dz, axis=(0,2,3))
    lwp = (lwc*dz).sum(dim='TH_1_60_eta_theta', skipna=True, min_count=1)
    # set some metadata for loading with iris later
    lwp = lwp.rename('liquid_water_path')
    lwp['rotated_latitude_longitude'] = rot_lat_lon
    lwp.grid_latitude_t.attrs['standard_name'] = 'grid_latitude'
    lwp.grid_longitude_t.attrs['standard_name'] = 'grid_longitude'
    # save output
    filename = f'data/model/{sim.code}_liquid_water_path_domain.nc'
    print(f'\n\n[INFO] saving to {filename}')
    lwp.to_netcdf(filename)

end = dt.datetime.now()
print(f'\n\n[END] {end-start}')