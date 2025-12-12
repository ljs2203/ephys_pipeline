#%% 
import os, sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
%load_ext autoreload
%autoreload 2

import bombcell as bc
# %%
# Replace with your kilosort directory
ks_dir = r'C:\Users\Josue Regalado\ephys_temp_data\NPX3\11_17_25_P1\finalcat_og\NPX3_11_17_25_training_CA_TH_g0\kilosort4_20251126_130031_CA1'

# Set bombcell's output directory
save_path = Path(ks_dir) / "bombcell"

print(f"Using kilosort directory: {ks_dir}")


# %%

## Provide raw and meta files
## Leave 'None' if no raw data. Ideally, your raw data is common-average-referenced and
# the channels are temporally aligned to each other (this can be done with CatGT)
#raw_file_path =  r"C:\Users\Josue Regalado\ephys_temp_data\NPX3\11_17_25_P1\finalcat_og\NPX3_11_17_25_training_CA_TH_g0\NPX3_11_17_25_training_CA_TH_g0_tcat.imec0.ap.bin" # "/home/jf5479/cup/Julie/from_Yunchang/20250411_4423_antibody_maze_C1/CatGT_out/catgt_20250411_4423_C1_g0/20250411_4423_C1_g0_imec0/20250411_4423_C1_g0_tcat.imec0.ap.bin" #None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0_bc_decompressed.imec0.ap.bin" # ks_dir
#meta_file_path = r"C:\Users\Josue Regalado\ephys_temp_data\NPX3\11_17_25_P1\finalcat_og\NPX3_11_17_25_training_CA_TH_g0\NPX3_11_17_25_training_CA_TH_g0_tcat.imec0.ap.meta" # "/home/jf5479/cup/Julie/from_Yunchang/20250411_4423_antibody_maze_C1/CatGT_out/catgt_20250411_4423_C1_g0/20250411_4423_C1_g0_imec0/20250411_4423_C1_g0_tcat.imec0.ap.meta" #None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0_bc_decompressed.imec0.ap.bin"None#"/home/julie/Dropbox/Example datatsets/JF093_2023-03-09_site1/site1/2023-03-09_JF093_g0_t0.imec0.ap.meta"
## Get default parameters - we will see later in the notebook how to assess and fine-tune these
meta_file_path = None
raw_file_path = None
param = bc.get_default_parameters(ks_dir, 
                                  raw_file=raw_file_path,
                                  meta_file=meta_file_path,
                                  kilosort_version=4)

print("Bombcell parameters:")
pprint(param)
# %%
# you might to change:

# 1. classification thresholds like: 


#  2. or which quality metrics are computed (by default these are not): 
param["computeDistanceMetrics"] = 0
param["computeDrift"] = 0
param["splitGoodAndMua_NonSomatic"] = 1 # splits good units into somatic and non somatic 
param['plotDetails'] = 1

#  3. how quality metricsa are calculated:
# a. how refractory period window is defined
param['minSpatialDecaySlopeExp'] = 0.005 
param['hillOrLlobetMethod'] = True # use Llobet et al correction to calculate refractory period violations
param['maxRPVviolations'] = 0.3
param["tauR_valuesMin"]= 1 / 1000  # minumum refractory period time (s), usually 0.002 s
# param["tauR_valuesMax"]= 5 / 1000  # maximum refractory period time (s)
# param["tauR_valuesStep"]= 1 / 1000  # if tauR_valuesMin and tauR_valuesMax are different, bombcell 
# # will calculate refractory period violations from param["tauR_valuesMin"] to param["tauR_valuesMax"] param["tauR_valuesStep"] 
# bins and determine the option window for each unit before calculating the violations. 
        # tauR_valuesStep
#  b. or whether the recording is split into time chunks to detemrine "good" time chunks: 
# param["computeTimeChunks"] = 0
# full list in the wiki or in the bc.get_default_parameters function
# %%
(
    quality_metrics,
    param,
    unit_type,
    unit_type_string,
) = bc.run_bombcell(
    ks_dir, save_path, param
)
# Use the output summary plots (below) to see if the 
# quality metric thresholds seem roughly OK for your 
# data (i.e. there isn't one threshold removing all 
# units or a threshold may below that removes none)
# more details on these output plots in the wiki:
# https://github.com/Julie-Fabre/bombcell/wiki/Summary-output-plots
# %%
del quality_metrics['maxChannels']
# quality metric values
quality_metrics_table = pd.DataFrame(quality_metrics)
quality_metrics_table.insert(0, 'Bombcell_unit_type', unit_type_string)
quality_metrics_table
# %%


# boolean table, if quality metrics pass threshold given parameters
boolean_quality_metrics_table = bc.make_qm_table(
    quality_metrics, param, unit_type_string
)
boolean_quality_metrics_table


# %%
# # Launch minimal GUI.
# # Ideally, take a look at your units for a few datasets so you can get an idea of which 
# # parameters will work best for your purposes. 
# gui = bc.unit_quality_gui(
#     ks_dir=ks_dir,
#     quality_metrics=quality_metrics,
#     unit_types=unit_type,
#     param=param,
#     save_path=save_path,
# )
# # %%
# bc.compare_manual_vs_bombcell(save_path)

# %%
