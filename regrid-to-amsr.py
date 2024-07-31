#!/usr/bin/env python3
# coding: utf-8
# regrid-to-amsr.py
# This script regrids model LWP field to AMSR2 horizontal grid.
# Also consolidates both datasets by extracting model domain from
# AMSR2 data and applying AMSR2 land/ice mask to model domain.
# iris package is used for regridding:
# https://scitools-iris.readthedocs.io/en/stable/
# =====================================================================
import iris
import numpy as np
import numpy.ma as ma
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import sys
sys.path.append("/home/users/eersp/cap_notebooks/")
import capricorn_figure_props as fprops
import capricorn_functions as cfn
import capricorn_constants as const
from simulations import simulations
from glob import glob
from importlib import reload

# ---------------------------------------------------------------------
def find_closest_time(cube, target_time):
    # For a given iris cube with a time dimension, return index of the
    # time point that's closest to target_time
    
    # Extract the time coordinate from the cube
    time_coord = cube.coord('time')
    
    # Convert target_time to a numeric value compatible with the time coordinate
    target_time_num = time_coord.units.date2num(target_time)
    
    # Find the nearest time point
    time_points = time_coord.points
    closest_time_index = np.abs(time_points - target_time_num).argmin()
    
    return closest_time_index

# ---------------------------------------------------------------------
start = dt.datetime.now()
iris.FUTURE.save_split_attrs = True  # suppress a warning
# ---------------------------------------------------------------------

# LOAD AMSR2 DATA
print('\n\n[INFO] Loading AMSR data', flush=True)
amsr_fn = glob('data/amsr2/RSS_AMSR2_ocean_L3_daily_2018-01-31_v08.2.nc')
print(f'[INFO] loading {amsr_fn}', flush=True)
amsr_lwp = iris.load_cube(amsr_fn, 'atmosphere_mass_content_of_cloud_liquid_water')
amsr_mask = amsr_lwp[0].data.mask  # 0 for data, 1 for ice/land

# ---------------------------------------------------------------------
# LOAD MODEL DATA
# Model data has been saved previously using compute-domain-lwp.py
print('\n\n[INFO] Loading model data', flush=True)
files = glob('data/model/*liquid_water_path_domain.nc')
sim_codes = [f.split('/')[2].split('_')[0] for f in files]
simulations = np.array([sim for sim in simulations if sim.code in sim_codes])
model_lwp = [iris.load_cube(file, 'liquid_water_path') for file in files]
print(f'\n\n[INFO] loaded {len(model_lwp)} simulations:', flush=True)
for sim in simulations:
    print(sim.code)

# ---------------------------------------------------------------------
# INTERPOLATE MODEL TO AMSR GRID
scheme = iris.analysis.Linear(extrapolation_mode='mask') 

# iris needs both cubes to have coordinate system before regridding
wgs84 = iris.coord_systems.GeogCS(
    semi_major_axis=6378137.0,
    inverse_flattening=298.257223563
)
amsr_lwp.coord('longitude').coord_system = wgs84
amsr_lwp.coord('latitude').coord_system = wgs84
rot_lat_lon = model_lwp[0].coord('rotated_latitude_longitude')
rot_cs = iris.coord_systems.RotatedGeogCS(
    grid_north_pole_latitude=rot_lat_lon.attributes['grid_north_pole_latitude'],
    grid_north_pole_longitude=rot_lat_lon.attributes['grid_north_pole_longitude']
)
for cube in model_lwp:
    cube.coord('grid_latitude').coord_system = rot_cs
    cube.coord('grid_longitude').coord_system = rot_cs

# call regrid
t0 = dt.datetime.now()
model_lwp_rgrd = iris.cube.CubeList([
    cube.regrid(amsr_lwp, scheme) for cube in model_lwp
])
t1 = dt.datetime.now()
print(f'\n\n[DEBUG] regridded all model output ({t1-t0})')

# ---------------------------------------------------------------------
# EXTRACT OVERPASS MODEL TIMESTEPS AND SAVE

# model_lwp_rgrd now has same shape as amsr_lwp in lat/lon dimensions
# but with datapoints outside of model domain masked
# combine this mask with AMSR land/sea mask
domain_mask = [cube[0].data.mask for cube in model_lwp_rgrd]
amsr_domain_mask = [(mask+amsr_mask).astype(bool) for mask in domain_mask]
model_lwp_rgrd.mask = amsr_domain_mask
# !!! this gives 2061 data points per 

# find nearest model timestep for each overpass time
# pass times defined 'by eye' by looking at orbit tracks
# on NASA worldview: https://go.nasa.gov/3sw3aT7 [31/08/2023]
overpass_times = [
    dt.datetime(2018,1,31,14,16), dt.datetime(2018,2,1,5,23),
    dt.datetime(2018,2,1,14,4),   dt.datetime(2018,2,2,5,6),
    dt.datetime(2018,2,2,14,47),  dt.datetime(2018,2,3,4,10),
    dt.datetime(2018,2,3,13,52),  dt.datetime(2018,2,4,4,54),
    dt.datetime(2018,2,4,14,33),  dt.datetime(2018,2,5,3,58),
]
print('\n\n[INFO] Saving regridded output')
for s,sim in enumerate(sim_codes):
    i_closest_time = [find_closest_time(model_lwp_rgrd[s], otime)
                      for otime in overpass_times]
    cube = model_lwp_rgrd[s][i_closest_time]
    mask_list = [amsr_domain_mask[s]]*cube.shape[0]
    mask = np.stack(mask_list,axis=0)
    cube.data.mask = mask
    # save to netCDF
    save_fn = f'data/model/{sim}_liquid_water_path_amsr.nc'
    print(f'[INFO] saving to {save_fn}')
    iris.save(cube, save_fn, fill_value=-99999.)

end = dt.datetime.now()
print(f'\n\n[END] {end-start}')