"""
simulation of collapse of transmission towers 
based on empricially derived conditional probability


required input
    -. tower geometry whether rectangle or squre (v) - all of them are square
    -. design wind speed (requiring new field in input) v
    -. tower type and associated collapse fragility v
    -. conditional probability by tower type (suspension and strainer) v
    -. idenfity strainer tower v

Todo:
    -. creating module (v)
    -. visualisation (arcGIS?) (v)
    -. postprocessing of mc results (v)
    -. think about how to sample random numbers (spatial correlation) (v)
    -. adding additional damage state (v)
    -. adj_list by function type (v)
"""

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap
from scipy import stats
import shapefile
import csv
import time
import cPickle as pickle
import simplekml
import pandas as pd
from scipy.optimize import minimize_scalar
'''

import numpy as np

#from compute import compute_
#import read
from read import (read_frag, read_cond_prob, read_tower_GIS_information, 
                 read_velocity_profile)

#import compute
from compute import (cal_collapse_of_towers_analytical,
    cal_collapse_of_towers_mc, cal_exp_std)

from class_Tower import Tower
from class_Event import Event

###############################################################################
# main procedure
###############################################################################

def main(shape_file_tower, shape_file_line, dir_wind_timeseries, 
    file_frag, file_cond_pc, file_design_value, file_terrain_height, 
    flag_strainer=None, flag_save=None, dir_output=None, nsims=100):

#unq_itime = None #np.array([900]) # or None
#unq_itime = np.array([22855.0]) # or 502 
#unq_itime = [502] # or 502 

#unq_itime_kml = [0, 502, 750]
#kml_output_file = 'pt3.kml'
#colorscheme = 'Reds'

    # read GIS information
    (tower, sel_lines, fid_by_line, fid2name, lon, lat) = \
    read_tower_GIS_information(Tower, shape_file_tower, shape_file_line, 
    file_design_value, file_terrain_height)

    # read collapse fragility by asset type
    (frag, ds_list, nds) = read_frag(file_frag)

    # read conditional collapse probability
    cond_pc = read_cond_prob(file_cond_pc)

    # calculate conditional collapse probability 
    for i in tower.keys():
        tower[i].idfy_adj_list(tower, fid2name, cond_pc, flag_strainer)
        tower[i].cal_cond_pc_adj(cond_pc, fid2name)

    # read wind profile and design wind speed
    event = read_velocity_profile(Event, dir_wind_timeseries, tower)
    idx_time = event[event.keys()[0]].wind.index
    ntime = len(idx_time)

    for i in tower.keys():
        event[i].cal_pc_wind(tower[i], frag, ntime, ds_list, nds) #
        event[i].cal_pc_adj(tower[i], cond_pc) #

    # analytical approach
    pc_collapse = {}
    for line in sel_lines:
        pc_collapse[line] = cal_collapse_of_towers_analytical(fid_by_line[line], 
            event, fid2name, ds_list)       
        if flag_save:
            for (ds, _) in ds_list:
                csv_file = dir_output + "/pc_line_" + ds + '_' + line.replace(' - ','_') + ".csv"
                pc_collapse[line][ds].to_csv(csv_file)
            
    print "Analytical calculation is completed"

    # mc approach
    # realisation of tower collapse in each simulation
    tf_sim = {} # dictionary of boolean array
    prob_sim = {} # dictionary of numerical array
    summary_line = {} # dictionary for expected and std of collapse

    for line in sel_lines:

        rv = np.random.random((nsims, ntime)) # perfect correlation within a single line

        for i in fid_by_line[line]:
            event[fid2name[i]].cal_mc_adj(tower[fid2name[i]], nsims, ntime, 
            ds_list, nds, rv)

        (tf_sim[line], prob_sim[line]) = (
        cal_collapse_of_towers_mc(fid_by_line[line], event, fid2name, ds_list, 
        nsims))

        summary_line[line]= cal_exp_std(tf_sim[line], ds_list, idx_time)

        if flag_save:

            for (ds, _) in ds_list:           
                npy_file = dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ','_') + ".npy"
                np.save(npy_file, tf_sim[line][ds])

                csv_file = dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                prob_sim[line][ds].to_csv(csv_file)

                csv_file = dir_output + "/summary_mc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                summary_line[line][ds].to_csv(csv_file)

    print "MC calculation is completed"
