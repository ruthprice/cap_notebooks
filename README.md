# cap_notebooks
This repo provides code and model data used to create the figures for Price et al., 2024, JGR (submitted).

July 2024

Contact Ruth Price, ruthpr@bas.ac.uk

<pre>
data/\
	model/\
	ERA5_init_XXX_cap.nc:			processed ERA5 data for figure 6\
	UKMO_init_XXX_cap.nc:			as above for UK Met Office analysis\
	u-XXXXX_mXXsXXiXX_CAP.nc:		time series of UM model output collocated with RV Investigator (spatial mean over 9 nearest neighbours). u-XXXXX is the UM code for the simulations (see eg simulations.py) and mXXsXXiXXX is the UM "stashcode" used in the model to reference each variable. Processed from raw output files with CDO and then collocated using capricorn-collocation-xr.py\
	u-XXXXX_mXXsXXiXX_grid_CAP.nc:		as above but 9x9 grid\
	u-XXXXX_liquid_water_path_amsr.nc:	as above regridded to AMSR2 spatial grid in regrid-to-amsr.py

capricorn_constants.py:		module for thermodyanmical and UM constants used in code\
capricorn_figure_props.py:	module for general definitions like fontsizes for figures\
capricorn_functions.py:		module for some utility functions used throughout \
era5_hybrid_coords.py:		module with some information on ERA5 coordinates\
simulations.py:			module containing a class with information about each UM simulation\
sonde.py:			module containing a class to read radiosonde data\
capricorn-collocate-xr.py:	script that collocates UM output with RV Investigator\
capricorn-collocate-init.py:	script that collocates UM output (at initilisation me), UKMO analysis and ERA5 with RV Investigator.\
compute-lwp-domain.py:		script to calculate and save gridded LWP from cloud liquid water mass, air density, and model level heights\
regrid-to-amsr.py:		script that uses the iris package to regrid model output to AMSR grid\
write-submission-files.py:	script that writes the files stashcodes.csv and suites.csv for later use in submit_cap_collocation.sh to submit collocation script to sbatch in bulk.\
compute_ukca_n10.ipynb:		notebook to calculate N10 variable from UKCA aerosol fields\
plot_amsr_lwp.ipynb:		notebook for figure 7\
plot_cloud_mask.ipynb:          notebook for figures S2 and S2\
plot_domain_maps.ipynb:		notebook for figure 1\
plot_lwp_rad_flux.ipynb:	notebook for figures 3, 4, 5, S4, S5 and S6\
plot_n10_cdnc: ipynb:		notebook for figures 2 and S1\
plot_theta_q_profiles.ipynb:	notebook for figures 6 and S7\
</pre>

This code was developed and used on the JASMIN computing facility using the jaspy software package. Latest version at time of writing (last accessed 31/07/2024):\
https://github.com/cedadev/ceda-jaspy-envs/blob/main/environments/r4.3/mf3-23.11.0-0/jasr4.3-mf3-23.11.0-0-r20240320/final-spec.yml
