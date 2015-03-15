'''
functions to compute collapse probability of tranmission towers

- pc_wind

- pc_adj_towers

- mc_adj

-  
'''

import numpy as np
import pandas as pd

def cal_collapse_of_towers_analytical(list_fid, event, fid2name, ds_list):
    """
    calculate collapse of towers analytically

    Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
    whre Pd: collapse due to direct wind
    Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))

    pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
    """

    ntower = len(list_fid)
    idx_time = event[event.keys()[0]].wind.index
    ntime = len(idx_time)
    cds_list = [x[0] for x in ds_list] # only string
    cds_list.remove('collapse') # non-collapse

    pc_adj_agg = np.zeros((ntower, ntower, ntime))
    pc_collapse = np.zeros((ntower, ntime))

    for irow, i in enumerate(list_fid):
        for j in event[fid2name[i]].pc_adj.keys():
            jcol = list_fid.index(j)
            pc_adj_agg[irow, jcol, :] = event[fid2name[i]].pc_adj[j]
        pc_adj_agg[irow, irow, :] = event[fid2name[i]].pc_wind.collapse.values

    pc_collapse = 1.0-np.prod(1-pc_adj_agg, axis=0) # (ntower, ntime)

    # prob of non-collapse damage
    prob = {}
    for ds in cds_list:

        temp = np.zeros_like(pc_collapse)
        for irow, i in enumerate(list_fid):

            val = (event[fid2name[i]].pc_wind[ds].values 
                            - event[fid2name[i]].pc_wind.collapse.values
                            + pc_collapse[irow, :])

            val = np.where(val > 1.0, 1.0, val)

            temp[irow, :] = val

        prob[ds] = pd.DataFrame(temp.T, columns = [fid2name[x] 
            for x in list_fid], index = idx_time)

    prob['collapse'] = pd.DataFrame(pc_collapse.T, columns = [fid2name[x] 
        for x in list_fid], index = idx_time)

    return prob


def cal_collapse_of_towers_mc(list_fid, event, fid2name, ds_list, nsims):

    ntower = len(list_fid)
    idx_time = event[event.keys()[0]].wind.index
    ntime = len(idx_time)
    cds_list = ds_list[:]
    cds_list.reverse() # [(collapse, 2), (minor, 1)]
    #cds_list = [x[0] for x in ds_list] # only string
    #cds_list.remove('collapse') # non-collapse
    #nds = len(ds_list)

    tf_ds = np.zeros((ntower, nsims, ntime), dtype=bool)

    for i in list_fid:

        # collapse by adjacent towers
        for j in event[fid2name[i]].mc_adj.keys(): # time

            for k in event[fid2name[i]].mc_adj[j].keys(): # fid

                isim = event[fid2name[i]].mc_adj[j][k]

                for l in k: # each fid

                    tf_ds[list_fid.index(l), isim, j] = True

        # collapse by wind
        #for (j1, k1) in zip(event[fid2name[i]].mc_wind['collapse']['isim'],
        #                    event[fid2name[i]].mc_wind['collapse']['itime']):
        #    tf_collapse_sim[list_fid.index(i), j1, k1] = True

                #if isinstance(k, tuple):
                #    for l in k: 
                #        tf_collapse_sim[list_fid.index(l), isim, j] = True

                #else:
                #    print "DDDD"
                #    tf_collapse_sim[list_fid.index(k), isim, j] = True       

                #try:
                #for l in k: # tuple
                #    tf_collapse_sim[list_fid.index(l), isim, j] = True
                #except TypeError: # integer
                #     tf_collapse_sim[list_fid.index(k), isim, j] = True
                # except IndexError:
                #     print 'IndexError %s' %i    

    #temp = np.sum(tf_collapse_sim, axis=1)/float(nsims)

    prob_sim, tf_sim = {}, {}

    #prob_sim['collapse'] = pd.DataFrame(temp.T, 
    #         columns = [fid2name[x] for x in list_fid], index = idx_time)

    # append damage stae by direct wind
    for (ds,_) in cds_list:

        for i in list_fid:

            for (j1, k1) in zip(event[fid2name[i]].mc_wind[ds]['isim'],
                                event[fid2name[i]].mc_wind[ds]['itime']):
                tf_ds[list_fid.index(i), j1, k1] = True

        temp = np.sum(tf_ds, axis=1)/float(nsims)
    
        prob_sim[ds] = pd.DataFrame(temp.T, 
            columns = [fid2name[x] for x in list_fid], index = idx_time)

        tf_sim[ds] = np.copy(tf_ds)

    #pc_collapse = np.zeros((ntower, ntime))

    # damage sate

    #for (ds, ids) in ds_list:

    #tf_collapse_sim = np.zeros((ntower, nsims, ntime), dtype=bool)

    #     temp = np.zeros((ntower, ntime))

    #     for irow, i in enumerate(list_fid):

    #         val = np.maximum(event[fid2name[i]].tf_wind, 
    #             tf_collapse_sim[irow,:,:]*nds) 

    #         temp[irow,:] = np.sum(val == ids+1, axis=0)/float(nsims)

    #     prob_sim[ds] = pd.DataFrame(temp.T, 
    #         columns = [fid2name[x] for x in list_fid], index = idx_time)

    return (tf_sim, prob_sim)

def cal_exp_std(tf_sim_line, ds_list, idx_time):
    """
    compute mean and std of no. of ds
    tf_collapse_sim.shape = (ntowers, nsim, ntime)
    """

    summary = {}

    for ds in tf_sim_line.keys():

        (ntowers, nsims, ntime) = tf_sim_line[ds].shape

        # mean and standard deviation
        x_ = np.array(range(ntowers+1))[:,np.newaxis] # (ntowers, 1)
        x2_= np.power(x_,2.0)

        no_ds_acr_towers = np.sum(tf_sim_line[ds],axis=0) #(nsims, ntime)
        no_freq = np.zeros((ntime, ntowers+1)) # (ntime, ntowers)

    #from scipy.stats import itemfreq
    #In [88]: freq_a = itemfreq(n_a)

        for i in range(ntime):
            for j in range(ntowers+1):
                no_freq[i, j] = np.sum(no_ds_acr_towers[:, i] == j)

        prob = no_freq / float(nsims) # (ntime, ntowers)

        exp_no = np.dot(prob,x_)

        std_no = np.power(np.dot(prob,x2_) - 
                          np.power(exp_no,2),0.5)

        summary[ds] = pd.DataFrame(np.hstack((exp_no, std_no)), 
            columns = ['mean', 'std'], index= idx_time)

    return summary
