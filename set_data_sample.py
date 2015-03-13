# script to set data for sim_towers_v13-2
shape_file_tower = './Shapefile_2015_01/Towers_NGCP_with_synthetic_attributes_WGS84.shp'
shape_file_line = './Shapefile_2015_01/Lines_NGCP_with_synthetic_attributes_WGS84.shp'
dir_wind_timeseries = './scenario50yr'

file_frag = 'fragility_dummy.csv'
file_cond_pc = 'cond_collapse_prob_dummy.csv'
file_design_value = 'design_value.csv'
file_terrain_height = 'terrain_height_multiplier.csv'

flag_strainer = ['Strainer','dummy'] # consider strainer 

# number of simulations for MC
nsims = 50

# flag for save
flag_save = 0

if __name__ == '__main__':
    from sim_towers_v13_2 import main
    main(shape_file_tower, shape_file_line, dir_wind_timeseries, 
    file_frag, file_cond_pc, file_design_value, file_terrain_height, 
    flag_strainer, flag_save, nsims)