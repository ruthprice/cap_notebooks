# capricorn_constants.py

# Contains constants used in CAPRICORN case study analysis
# RP Sep 2023
# ======================================================================
# ----------------------------------------------------------------------
p0 = 100000.0 # reference pressure, Pa
Rd = 287.05   # sp. gas const. for dry air, J/kg/K
cp = 1005.46  # sp. heat capacity of air at const. pressure, J/kg/K
Rd_cp = Rd/cp
Rd = 287.058
molar_mass_air = 28.991e-3 # kg per mole
molar_mass_water = 18.0e-3 # kg per mole
avogadro_number = 6.022e23 # particles per mole

mode_sig = { # UKCA mode widths
    'soluble_nucleation': 1.59,
    'soluble_aitken':1.59,
    'soluble_accumulation': 1.4,
    'soluble_coarse': 2.0,
    'insoluble_aitken': 1.59,
    }

ztop = 40000.0  # altitude of model top
