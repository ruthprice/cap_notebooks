# capricron-figure-props.py
# Contains settings to control figures for CAPRICORN case study
# RP Sep 2023
# =====================================================================

# Simulation names for legends
suite_labels = {
    'u-cr700': 'CONTROL',
    'u-cu657': 'CONTROL',
    'u-ct323': 'FIXED_DROP',
    'u-cw067': 'LBC_CASIM_ICE',
    'u-cw894': 'ERA5_IC',
    'u-dc034': 'BL_CU_DIAG',
    'u-dd355': 'ALL_TESTS'
}

# Colours for lines etc in graphs
colours = {
    'u-cr700': 'tab:blue',
    'u-cu657': 'tab:blue',
    'u-ct323': 'tab:orange',
    'u-cw067': 'tab:orange',
    'u-cw894': 'tab:green',
    'u-dc034': 'tab:green',
    'u-dd355': 'tab:red'
}

# Linestyles for graphs
thin = 0.75
thick = 1.2
linewidths = {
    'u-cr700': thick,
    'u-cu657': thick,
    'u-ct323': thick,
    'u-cw067': thin,
    'u-cw894': thick,
    'u-dc034': thin,
    'u-dd355': thick
}

# fontsizes
ax_fs = 8
ax_label_fs = 10
label_fs = 10
legend_fs = 5

dpi = 300      # dots per sq. inch
in_cm = 1/2.54    # converts inches to cm for setting figure sizes

