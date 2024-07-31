# simulations.py
# Script containing class and defintions of UM simulations
# for UM suite codes, figure labels, etc
# =====================================================================
import numpy as np

class Simulation:
    # A class to contain information about each simulation,
    # including eg its UM suite and configuration name,
    # or what colour to use when plotting data.
    def __init__(self, code, config, label, colour, linewidth):
        self.code = code
        self.config = config
        self.label = label
        self.colour = colour
        self.linewidth = linewidth

# ---------------------------------------------------------------------
simulations = np.array([
    Simulation('u-cr700', '', 'CONTROL', 'tab:blue', 1.2),
    Simulation('u-cu657', '_ra3_ukca_casim', 'CONTROL', 'tab:blue', 1.2),
    Simulation('u-ct323', '_fixed_drop_150cc', 'FIXED_DROP', '#ff7f00', 1.2),
    Simulation('u-cw067', '_RAL3p1_ukca', 'ICE_MICROPHYS', 'tab:orange', 0.75),
    Simulation('u-cw894', '_RAL3p1_ukca', 'ERA5_INITIAL', 'tab:green', 1.2),
    Simulation('u-dc034', '_RAL3p1_ukca', 'MIXING', 'tab:green', 0.75),
    Simulation('u-dd355', '_RAL3p1_ukca', 'ALL_TESTS', 'tab:red', 1.2)
])