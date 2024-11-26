#!/usr/bin/env python3
# coding: utf-8
# capricorn-collocate-init.py
# code to collocate model initialisation fields (UKMO analysis, ERA5
# and domain output from first archived timestep after initialisation)

# RP July 2024
# ======================================================================
import numpy as np
import iris
import iris.cube
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.append("/home/users/eersp/cap_notebooks/")
from capricorn_functions import get_cap_coords
import capricorn_constants as const
from era5_hybrid_coords import a as era_a
from era5_hybrid_coords import b as era_b
from simulations import simulations

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

dates = [
    dt.datetime(2018,1,22),
    dt.datetime(2018,1,23),
    dt.datetime(2018,1,24),
    dt.datetime(2018,1,25),
    dt.datetime(2018,1,26),
    dt.datetime(2018,1,27),    
    dt.datetime(2018,1,31),
    dt.datetime(2018,2,1),
    dt.datetime(2018,2,2),
    dt.datetime(2018,2,3),
    dt.datetime(2018,2,4),
]
dump_output_dir = '/gws/nopw/j04/asci/rprice/start_dump/'

# load UK Met Office global analysis
print('\n\n[INFO] loading UKMO data', flush=True)
ukmo_fmt = '%Y%m%dT%H%MZ'
ukmo_dates = [dt.datetime.strftime(T, ukmo_fmt) for T in dates]
ukmo_files = [glob(f'{dump_output_dir}{T}_glm_t+0')[0] for T in ukmo_dates]
ukmo_pot_temp = iris.cube.CubeList([])
ukmo_humidity = iris.cube.CubeList([])
t0 = dt.datetime.now()
for fn in ukmo_files:
    print(f'[INFO] loading {fn}', flush=True)
    cubes = iris.load(fn)
    for cube in cubes:
        if cube.attributes['STASH'] == 'm01s00i004':
            ukmo_pot_temp.append(cube)
        elif cube.attributes['STASH'] == 'm01s00i010':
            ukmo_humidity.append(cube)
ukmo_pot_temp = ukmo_pot_temp.merge_cube()
ukmo_humidity = ukmo_humidity.merge_cube()
t1 = dt.datetime.now()
print(f'\n\n[DEBUG] finished loading UKMO files ({t1-t0})')
# get lat/lons
if not ukmo_pot_temp.coord('longitude').has_bounds():
    ukmo_pot_temp.coord('longitude').guess_bounds()
if not ukmo_pot_temp.coord('latitude').has_bounds():
    ukmo_pot_temp.coord('latitude').guess_bounds()
ukmo_lon_bounds = np.unique(ukmo_pot_temp.coord('longitude').bounds.flatten())
ukmo_lat_bounds = np.unique(ukmo_pot_temp.coord('latitude').bounds.flatten())

# Load ERA5
print('\n\n[INFO] Loading ERA5 grib files')
era_fmt = '%Y%m%d_%H'
era_dates = [dt.datetime.strftime(T, era_fmt) for T in dates]
era_files = [glob(f'{dump_output_dir}ec_grib_{T}.grib')[0] for T in era_dates]
era_data = []
t0 = dt.datetime.now()
for fn in era_files:
    print(f'[INFO] loading {fn}', flush=True)
    # ERA5 grib files are mix of grib1 and grib2 which xr doesn't like
    # load them separately (https://github.com/ecmwf/cfgrib/issues/83
    # last accessed 14/05/24)
    kwarg = [{'filter_by_keys':{'edition': n}} for n in [1,2]]
    era_data1 = xr.open_dataset(fn, engine='cfgrib',
                                backend_kwargs=kwarg[0])
    era_data2 = xr.open_dataset(fn, engine='cfgrib',
                                backend_kwargs=kwarg[1])
    era_merged = xr.merge([era_data1, era_data2])
    era_data.append(era_merged)
era_data = xr.concat(era_data, dim='time')
t1 = dt.datetime.now()
print(f'\n\n[DEBUG] finished loading ERA5 files ({t1-t0})')

# calculate era5 geometric altitude from pressure levels
era_k = era_data.hybrid.values
era_sp_4d = np.expand_dims(era_data.sp.values, axis=1)
era_a_4d = np.expand_dims(era_a[1:], axis=[0,2,3])
era_b_4d = np.expand_dims(era_b[1:], axis=[0,2,3])
shape = era_data.t.shape
shape = (shape[0], shape[1]+1, shape[2], shape[3])
# pressure on half levels
era_ph = np.zeros((shape))
era_ph[:,1:] = era_a_4d + era_sp_4d*era_b_4d
era_ph[:,0] = 0
era_dp = era_ph[:,1:] - era_ph[:,:-1]
# pressure on full levels
era_pf = 0.5*(era_ph[:,1:] + era_ph[:,:-1])
era_rho = era_pf / (era_data.t.values * const.Rd)
g = 9.81 # m s-1
era_dz = era_dp / (g*era_rho)
era_z = np.cumsum(era_dz[:,::-1], axis=1)[:,::-1]
era_pot_temp = era_data.t.values * (const.p0/era_pf)**const.Rd_cp
dims = ['time','hybrid','latitude','longitude']
era_data['altitude'] = (dims, era_z)
era_data['pot_temp'] = (dims, era_pot_temp)

# Load model output
print('[info] loading CAPRICORN simulations')
sim_codes = ['u-dc034']
simulations = [sim for sim in simulations if sim.code in sim_codes]
sim_dates = np.array([
    # matrix for which simulations run on which dates
    [0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1],
]).astype(bool)
model_output_dir = [f'/gws/nopw/j04/asci/rprice/ukca_casim_output/{sim.code}/'
                  for sim in simulations]

output_files = []
for s,sim in enumerate(simulations):
    flist = [glob(f'{model_output_dir[s]}{T}{sim.config}_umnsaa_pa000.nc')[0]
             for T in np.array(ukmo_dates)[sim_dates[s]]]
    output_files.append(flist)
stashcodes = ['STASH_m01s00i004','STASH_m01s00i010','rotated_latitude_longitude']

# for all simulations, load first output file of each forecast cycle
# extract potential temperature and vapour stash fields
# and take first element in time dimension
model_data = [xr.concat([xr.open_dataset(fn)[stashcodes].isel(T1HR=0)
              for fn in flist], dim='T1HR') for flist in output_files]

# ---------------------------------------------------------------------
# EXTRACT SHIP LOCATION FROM MODEL FIELDS

# for each element of dates, get index of cap_times at that time
i_Tinit_to_Tship = np.array([get_closest_time(T, cap_times)
                             for T in dates])
# take subset of CAPRICORN coords that are at model timesteps
cap_lons = cap_lons[i_Tinit_to_Tship]
cap_lats = cap_lats[i_Tinit_to_Tship]

# UKMO
i_ukmo = np.digitize(cap_lons, ukmo_lon_bounds)-1
j_ukmo = np.digitize(cap_lats, ukmo_lat_bounds)-1
ukmo_pot_temp_ship = iris.cube.CubeList([ukmo_pot_temp[t,:,j_ukmo[t],i_ukmo[t]]
                                         for t in np.arange(len(dates))]).merge_cube()
ukmo_humidity_ship = iris.cube.CubeList([ukmo_humidity[t,:,j_ukmo[t],i_ukmo[t]]
                                         for t in np.arange(len(dates))]).merge_cube()

# ERA5
era_data_ship = xr.concat([era_data.isel(time=t).sel(latitude=cap_lats[t],
                                                     longitude=cap_lons[t], method='nearest')
                           for t in np.arange(len(dates))], dim='time')

# UM
PC = ccrs.PlateCarree()
model_data_ship = []
dates = np.array(dates)
for s,sim in enumerate(simulations):
    i = sim_dates[s]
    # get CAP coords on model rotated pole projection
    rot_coords = model_data[s].rotated_latitude_longitude
    rot_lon = rot_coords.grid_north_pole_longitude
    rot_lat = rot_coords.grid_north_pole_latitude
    rot_pole = ccrs.RotatedPole(
        pole_latitude=rot_lat,
        pole_longitude=rot_lon
    )
    cap_rlons = rot_pole.transform_points(src_crs=PC, x=cap_lons[i], y=cap_lats[i])[:,0]
    cap_rlons[cap_rlons<0] += 360
    cap_rlons[cap_rlons>360] -= 360
    cap_rlats = rot_pole.transform_points(src_crs=PC, x=cap_lons[i], y=cap_lats[i])[:,1]
    # use rotated coords in xr.ds.sel()
    ds = [
        model_data[s].sel(
            T1HR=time,
            grid_longitude_t=cap_rlons[t],
            grid_latitude_t=cap_rlats[t],
            method="nearest"
        ) for t,time in enumerate(dates[sim_dates[s]])
    ]
    ds = xr.concat(ds, dim='T1HR')
    print(s,sim.code,ds)
    model_data_ship.append(ds)

# ---------------------------------------------------------------------
# PLOT MAP OF EXTRACTED COORDINATES
plot = False
if plot:
    fig = plt.figure(figsize=(10,10), dpi=300)
    axes = plt.axes(projection=ccrs.SouthPolarStereo())
    axes.set_extent([120,180,-55,-75], crs=PC)
    axes.coastlines()
    axes.plot(cap_lons, cap_lats, 'k.', ms=2, label='CAP', transform=PC)
    axes.plot(ukmo_humidity_ship.coord('longitude').points,
              ukmo_humidity_ship.coord('latitude').points, marker='s',
              ls='None', ms=2, transform=PC, label='UKMO')
    axes.plot(era_data_ship.longitude.values,
              era_data_ship.latitude.values, marker='^', ls='None',
              ms=2, transform=PC, label='ERA')
    for s,sim in enumerate(simulations):
        axes.plot(model_data_ship[s].longitude_t.values,
                  model_data_ship[s].latitude_t.values, marker='<',
                  ls='None', ms=2, transform=PC, label=sim.label)
    plt.legend(fontsize=6)
    plt.show()

# ---------------------------------------------------------------------
# SAVE 
print('\n\n[INFO] saving the profiles to file')
save_dir = 'data/model/'
ukmo_pt_fn = 'UKMO_init_theta_cap.nc'
ukmo_q_fn = 'UKMO_init_humidity_cap.nc'
iris.save(ukmo_pot_temp_ship, save_dir+ukmo_pt_fn)
iris.save(ukmo_humidity_ship, save_dir+ukmo_q_fn)
era_pt_fn = 'ERA5_init_theta_cap.nc'
era_q_fn = 'ERA5_init_humidity_cap.nc'
era_data_ship[['pot_temp','altitude']].to_netcdf(save_dir+era_pt_fn)
era_data_ship[['q','altitude']].to_netcdf(save_dir+era_q_fn)
for s,sim in enumerate(simulations):
    sim_pt_fn = f'{sim.code}_init_theta_cap.nc'
    sim_q_fn = f'{sim.code}_init_humidity_cap.nc'
    model_data_ship[s]['STASH_m01s00i004'].to_netcdf(save_dir+sim_pt_fn)
    model_data_ship[s]['STASH_m01s00i010'].to_netcdf(save_dir+sim_q_fn)

# ======================================================================
end = dt.datetime.now()
print(f'\n\n[END] finished in {end-start}')
