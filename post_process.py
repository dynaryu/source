# post-processing of simulation results
# 0. comparision between two approaches
# 2. plot exp and std
# 3. No. of collapse vs. Prob (with and without cascading effect)
# 4. Prob. of no. of collapse at the max.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap
#from scipy import stats
#import shapefile
#import csv
#import time
#import cPickle as pickle
#import simplekml

from read import read_design_value, read_frag

def read_output(dir_output, sel_lines, ds_list, file_str, flag):

    output={}    
    for line in sel_lines:
        for (ds, _) in ds_list:
            if flag == 'csv':
                csv_file = dir_output + file_str + ds + '_' + line.replace(' - ','_') + ".csv"
                tmp = pd.read_csv(csv_file, index_col=0, parse_dates=[0])
            elif flag == 'npy':
                npy_file = dir_output + file_str + ds + '_' + line.replace(' - ','_') + ".npy"
                tmp = np.load(npy_file)    
            output.setdefault(line,{})[ds] = tmp
    return output

def plot_exp_std(summary, sel_lines, year_str, title_str=None, time_win=None, 
    flag_save=None):

    for line in sel_lines:
        for (ds, _) in ds_list:

            plt.figure()
            summary[line][ds]['mean'].plot(style='r-',xlim=time_win)
            summary[line][ds]['std'].plot(style='b--',xlim=time_win)
            
            plt.xlabel('Time')
            plt.ylabel('No. of tower')
            plt.title(line + ':' + ds + ':' + year_str + title_str)
            plt.legend(['Mean','Std'])

            if flag_save:
                plt.savefig('est_ntower_' + line+'_'+ds+'_'+ year_str + title_str + '.png')

def plot_bar_ntower(prob_ntower, est_ntower, year_str, saveName=None, flag_save=None):

    value = {}
    for line in prob_ntower.keys():

        #for ds in prob_ntower[line].keys():
        idx = np.argmax(est_ntower[line]['collapse']['mean'])

        prob_array = prob_ntower[line]['collapse'][idx,:]

        value[line] = prob_array
        nrow = len(prob_array)
        plt.figure()
        plt.stem(np.arange(nrow), prob_array)

        plt.xlabel('No. of collapsed tower')
        plt.ylabel('Probability')
        plt.title(line + ': '+year_str)

        if flag_save:
            plt.savefig(saveName + line + '_collapse.png')

    return value

def summary_table(est_ntower, sel_lines, ds_list):

    value = {}
    for line in sel_lines:
        for (ds, _) in ds_list:
            tmp = est_ntower[line][ds]['mean'].max()
            value.setdefault(line,{})[ds] = np.round(tmp)
    return value

def plot_surface(prob, xlab, ylab, zlab, saveName=None, flag_save=None):

    #import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    #import random

    for line in prob.keys():
        for ds in prob[line].keys():

            if isinstance(prob[line][ds],np.ndarray):
                prob_array = prob[line][ds]
            else:    
                prob_array = prob[line][ds].as_matrix()

            (nx, ny) = prob_array.shape
            x = np.arange(0, nx, 1)
            y = np.arange(0, ny, 1)
            X, Y = np.meshgrid(x, y)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, prob_array.T, cmap=cm.coolwarm, linewidth=0.0)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_zlabel(zlab + ':' + ds)
            ax.set_title(line)

            if flag_save:
                plt.savefig(saveName + line +'_'+ds + '.png')            
            # dd = [item.get_text() for item in ax.get_yticklabels()]
            # print dd
            # plt.show()

            # ylabels = 

            # nylabels = []
            # for item in ylabels:
            #     try:
            #         nylabels.append(tower_list[int(item)])
            #     except IndexError:
            #         nylabels.append('')   

            # ax.set_yticklabels(nylabels)

def check_wind_velocity():
    '''
     dummy 
    '''
    max_wind_speed  = {}
    max_dir_wind_speed  = {}
    for line in sel_lines:
        tmp_a, tmp_b, tmp_c = [], [], []
        for i in fid_by_line[line]:
            a = event[fid2name[i]].wind.speed.max()
            b = event[fid2name[i]].wind.dir_speed.max()
            c = event[fid2name[i]].vratio
            tmp_a.append(a)
            tmp_b.append(b)
            tmp_c.append(c)
        max_wind_speed[line] = np.array(tmp_a)
        max_dir_wind_speed[line] = np.array(tmp_b)

    adj_capacity, value  = {}, {}
    for line in sel_lines:
        tmp, tmp1 = [], []
        for i in fid_by_line[line]:
            a = tower[fid2name[i]].adj_design_speed
            tmp.append(a)
            tmp1.append(tower[fid2name[i]].actual_span)
        adj_capacity[line] = np.array(tmp)
        value[line] = np.array(tmp1)

    for line in max_wind_speed.keys():
        plt.figure();
        nt = len(max_wind_speed[line])
        plt.plot(range(nt), max_wind_speed[line], 'b-',
            range(nt), max_dir_wind_speed[line], 'g--',
            range(nt), adj_capacity[line], 'r-')
        plt.title(line)
        plt.legend(['max wind', 'max adj wind', 'adj capacity'])


    line_dic = {}    
    for line in sel_lines:
        tmp_suspension, tmp_strainer, tmp_angle = [], [], []

        for i in fid_by_line[line]:

            a = tower[fid2name[i]].u_val

            if tower[fid2name[i]].funct == 'Strainer':
                tmp_strainer.append(a)
                tmp_angle.append(tower[fid2name[i]].dev_angle)
            else:
                tmp_suspension.append(a)    

        line_dic.setdefault(line, {})['suspension'] = tmp_suspension
        line_dic.setdefault(line, {})['angle'] = tmp_angle
        line_dic.setdefault(line, {})['strainer'] = tmp_strainer

        tmp_b.append(b)
        tmp_c.append(c)
        max_wind_speed[line] = np.array(tmp_a)
        max_dir_wind_speed[line] = np.array(tmp_b)


###############################################################################

# main 
pdir = '/Users/hyeuk/Project/infrastructure/transmission'

file_frag = pdir + '/input/fragility_dummy.csv'
file_design_value = pdir + '/input/design_value_50yr.csv'

(sel_lines, _) = read_design_value(file_design_value)
sel_lines.remove('Amadeo - Dasmarinas')
sel_lines.remove('Santa Rosa - Binan')
sel_lines.remove('Santa Rita - Batangas')

(_, ds_list, nds) = read_frag(file_frag)


#output_dir = '/output_s50yr_d50yr/'
output_dir = '/output50yr_GA/'
est_ntower50_nc = read_output(pdir + output_dir, sel_lines, ds_list, 'est_ntower_nc_','csv')
est_ntower50 = read_output(pdir + output_dir, sel_lines, ds_list, 'est_ntower_','csv')

print summary_table(est_ntower50_nc, sel_lines, ds_list)

print summary_table(est_ntower50, sel_lines, ds_list)

"""
# read simulation output
file_str = '/est_ntower_nc_'
est_ntower50_nc = read_output(pdir + '/output50yr_NGCP/', sel_lines, ds_list, file_str,'csv')
est_ntower100_nc = read_output(pdir + '/output100yr_NGCP/', sel_lines, ds_list, file_str,'csv')
est_ntower200_nc = read_output(pdir + '/output200yr_NGCP/', sel_lines, ds_list, file_str,'csv')
#est_glenda_nc = read_output(pdir + '/output_glenda/', sel_lines, ds_list, file_str,'csv')

file_str = 'est_ntower_'
est_ntower50 = read_output(pdir + '/output50yr_NGCP/', sel_lines, ds_list, file_str,'csv')
est_ntower100 = read_output(pdir + '/output100yr_NGCP/', sel_lines, ds_list, file_str,'csv')
est_ntower200 = read_output(pdir + '/output200yr_NGCP/', sel_lines, ds_list, file_str,'csv')
est_glenda = read_output(pdir + '/output_glenda/', sel_lines, ds_list, file_str,'csv')

file_str = '/prob_ntower_nc_'
prob_ntower50_nc = read_output(pdir + '/output50yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower100_nc = read_output(pdir + '/output100yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower200_nc = read_output(pdir + '/output200yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower_glenda_nc = read_output(pdir + '/output_glenda/', sel_lines, ds_list, file_str,'npy')

file_str = '/prob_ntower_'
prob_ntower50 = read_output(pdir + '/output50yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower100 = read_output(pdir + '/output100yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower200 = read_output(pdir + '/output200yr_NGCP/', sel_lines, ds_list, file_str,'npy')
prob_ntower_glenda = read_output(pdir + '/output_glenda/', sel_lines, ds_list, file_str,'npy')

# save plot for all cases

#time_win = ['2014-07-15 18:00:00','2014-07-16 00:00:00']
#plot_exp_std(est_glenda_nc, sel_lines, 'TY Glenda', ' without cascade', time_win, 1)
#plot_exp_std(est_glenda, sel_lines, 'TY Glenda', ' with cascade', time_win, 1)

time_win = ['2014-07-15 21:00:00','2014-07-16 03:00:00']
plot_exp_std(est_ntower50, sel_lines, '50 year', ' with cascade', time_win, 1)
plot_exp_std(est_ntower50_nc, sel_lines, '50 year', ' without cascade', time_win, 1)

# for selected line comparision among three return periods

# without cascading effect
plt.figure()
line = 'Calaca - Santa Rosa'
time_win = ['2014-07-15 21:00:00','2014-07-16 03:00:00']
est_ntower50_nc[line]['collapse']['mean'].plot(style='b-',xlim=time_win)
est_ntower100_nc[line]['collapse']['mean'].plot(style='g-',xlim=time_win)
est_ntower200_nc[line]['collapse']['mean'].plot(style='r-',xlim=time_win)
plt.legend(['50 year','100 year','200 year'])

# glenda case
est_glenda = read_output(pdir + '/output_glenda', sel_lines, ds_list, '/est_ntower_','csv')

plot_exp_std(summary, year_str
plot_exp_std(est_glenda_nc, 'TY Glenda')

plot_exp_std(est_glenda, 'TY Glenda')

#plt.figure()
#line = 'Calaca - Santa Rosa'
#time_win = ['2014-07-15 21:00:00','2014-07-16 03:00:00']
#est_glenda_nc[line]['collapse']['mean'].plot(style='b-')


# plot prob of no. of tower along time

prob_max_ntower50_nc = plot_bar_ntower(prob_ntower50_nc, est_ntower50_nc, '50 year')
prob_max_ntower100_nc = plot_bar_ntower(prob_ntower100_nc, est_ntower100_nc, '100 year')
prob_max_ntower200_nc = plot_bar_ntower(prob_ntower200_nc, est_ntower200_nc, '200 year')



# # plot
# for line in sel_lines:
#     for (ds, _) in ds_list:
#         plt.figure()
#         for item in prob_anl[line][ds].keys(): 
#             prob_anl[line][ds][item].plot(xlim=['2014-07-15 12:00:00','2014-07-16 12:00:00'])

#for line in prob_anl.keys():
#    print 'number of towers for %s is %s' %(line, prob_anl[line]['collapse'].shape[1])


def comparision_between_methods(prob_sim, prob_anl, time_stamp):

    diff = []
    line = 'Batangas - Makban'
    ds = 'collapse'
    #for line in prob_sim.keys():
        #for ds in prob_sim[line].keys():
    for twr in prob_sim[line][ds].keys(): # tower
        dd = np.abs(prob_anl[line][ds][twr]-prob_sim[line][ds][twr]).max()
        diff.append(dd)

        plt.figure()
        prob_anl[line][ds][twr].plot(style='b-',xlim=['2014-07-15 12:00:00','2014-07-16 12:00:00'])
        prob_sim[line][ds][twr].plot(style='ro',xlim=['2014-07-15 12:00:00','2014-07-16 12:00:00'])

        plt.savefig('./temp2/comp_'+twr + '.png')
        plt.close()


        time_max = np.argmax(exp_no)
        summary_time_max = np.vstack(range(ntowers+1), prob[time_max, :])


# export to kml
#export_to_kml(kml_output_file, colorscheme, unq_itime_kml)

# visualisation ?? GIS one snapshot vs. temporal visulaisation??
# Need to talk to Matt
# Alternatively I can generate using python matplotlib Basemap

#plt.figure()
"""

'''

plt.xlim([22600, 23200])

plt.figure()
plt.plot(pc_collapse[fid_by_line[sel_line],451:571])

for j in range(500, 520):
    plt.figure()
    plt.plot(fid_by_line[sel_line],pc_collapse[fid_by_line[sel_line],j],
        fid_by_line[sel_line], pc_collapse_sim[sel_line][:,j],'ro')
    #plt.ylim([0, 1])

mp = summary_collapse_line[sel_line]['mean']+summary_collapse_line[sel_line]['std']

mm = summary_collapse_line[sel_line]['mean']-summary_collapse_line[sel_line]['std']

mm[mm<=0] = 0.0

plt.figure()
plt.plot(towers[320].wind['time'],
    summary_collapse_line[sel_line]['mean'],'r-',
    towers[320].wind['time'],    
    mp,'b--',
    towers[320].wind['time'],
    mm,'g--')


#plt.show()

#plt.savefig('compare.png')
'''


"""
def draw_map(figsize_tuple, lon, lat, val_array):

#map = Basemap(projection='merc', resolution = 'h', area_thresh = 0.1,
#    llcrnrlon=120.5, llcrnrlat=13.7,
#    urcrnrlon=121.5, urcrnrlat=14.7)

    if figsize_tuple == None:
        figsize_tuple = (8, 8)

    (ntwr, ntime) = val_array.shape

    #fig = plt.figure(figsize=figsize_tuple)

    plt.figure()
#    map = Basemap(projection='merc', resolution = 'h', area_thresh = 0.1,
#        llcrnrlon=120.5, llcrnrlat=13.75,
#        urcrnrlon=121.25, urcrnrlat=14.75)

    map = Basemap(projection='merc', resolution = 'h', area_thresh = 0.1,
        llcrnrlon=121.0, llcrnrlat=14.25,
        urcrnrlon=121.1, urcrnrlat=14.35)
      
    map.drawcoastlines()


    # labels = [left,right,top,bottom]
    parallels = np.arange(0.,90,0.25)
    map.drawparallels(parallels,labels=[False,True,True,False])

    meridians = np.arange(10.,360.,0.25)
    map.drawmeridians(meridians,labels=[True,False,False,True])


    #x, y = map(lon, lat)

    x, y = map(lon[fid_by_line[sel_line]], lat[fid_by_line[sel_line]])

    #max_r = 5.0


    val = val_array[fid_by_line[sel_line], 502]

    #map.scatter(x, y, c=val, s = np.pi*(max_r *val)**2, cmap=cm.hot_r, edgecolors = 'none')
    #map.scatter(x, y, c=val, s = np.pi*(max_r *val)**2, 
    #    cmap=cm.jet, edgecolors = 'none')
    #cbar = map.colorbar(None,location='bottom',pad="5%")
    #cbar.set_label('Collapse probability')

    map.scatter(x, y, c=val, s = 10.0, 
        cmap=cm.jet, edgecolors = 'none')

    cbar = map.colorbar(cs,location='bottom',pad="5%")

    plt.savefig('trial_0.eps')

    for j in range(1, ntime):
        val = val_array[:, j]
        plot_handle.set_array(val)
        plt.savefig('trial_' + str(j) +'.png')


# add colorbar.

plt.show()

def get_list_cascading_collapse(xth,ntowers,idx_adj_towers,tf_w0,rv0):

    sidx = idx_adj_towers[(idx_adj_towers+xth>=0) & (idx_adj_towers+xth < ntowers)]

    idx_list = [] #i, j, prob
    ith = np.intersect1d(sidx,pc_n.keys())
    for i in ith:
        jth = np.intersect1d(sidx,pc_n[i].keys())
        for j in jth:
                idx_list.append((i,j,pc_n[i][j]))

    '''            
    # cut-off in case of alive strain tower
    dd_idx_list = []
    for i in sidx:
        #print i, tower_dic[i+xth+1]['type'], tf_w0[i+xth]
        if (tower_dic[i+xth+1]['type'] in selected_tower_type) & (tf_w0[i+xth] == False):
            for jj in idx_list:
                #print jj[0], jj[1]
                if (jj[0] <= i) and (jj[1] >= i):
                    dd_idx_list.append(jj)
                    #print jj[0], jj[1], 'Removed' #idx_list.remove(jj)
    '''

    dd_idx_list = []                
    valid_idx_list = list(set(idx_list)-set(dd_idx_list))                    

    # convert cumulative prob
    cum_prob = []
    prob = 0.0
    for line in valid_idx_list:
        prob += line[2]
        cum_prob.append(prob)

    id_tf = np.sum(rv0 > cum_prob) # 

    if id_tf < len(valid_idx_list):
        res = range(valid_idx_list[id_tf][0]+xth, valid_idx_list[id_tf][1]+xth+1)
    else:    
        res = []

    return res


def get_line_type(no_lines):
    line_type = {}
    i = 0
    for l in ['-','--','-.',':']:
        for c in ['b','g','r','c']:
            line_type[str(i)] = c + l 
            line_type[str(i)] = c + l 
            i += 1
            if i > no_lines:
                break
    return line_type

def sort_by_nc(dic_a):
    dic_summary = {}
    for i in dic_a.keys():
        temp = []
        for j in dic_a[i].keys():
            temp.append(dic_a[i][j])
        dic_summary[i] = sum(temp)

    sorted_i = sorted(dic_summary,key=dic_summary.get)
    sorted_i.reverse()

    temp = []
    for i in sorted_i:
        temp.append((i,dic_summary[i]))

    return temp

def sort_by_the_most_common_pattern_given_nc(dic_a,i):
    sorted_i = sorted(dic_a[i],key=dic_a[i].get)
    sorted_i.reverse()
    
    temp = []
    for j in sorted_i:
        temp.append((j,dic_a[i][j]))

    return temp

def sort_by_the_most_common_pattern_for_each_nc(dic_a):
    max_temp = {}
    for i in dic_a.keys():
        temp = pick_the_most_common_pattern_out_of_nc(dic_a,i)
        j = 0
        for line in temp:
            if line[1] >= j:
                max_temp[line[0]] = line[1]
                j = line[1]
    
    sorted_i = sorted(max_temp,key=max_temp.get)
    sorted_i.reverse()

    temp = []
    for i in sorted_i:
        temp.append((i,max_temp[i]))

    return temp    


def sort_by_the_most_common_pattern(dic_a):
    dic_summary = {}
    for i in dic_a.keys():
        for j in dic_a[i].keys():
            dic_summary[j] = dic_a[i][j]

    sorted_i = sorted(dic_summary,key=dic_summary.get)
    sorted_i.reverse()

    temp = []
    for i in sorted_i:
        temp.append((i,dic_summary[i]))
    return temp    
"""



