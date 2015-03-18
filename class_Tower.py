import numpy as np
import copy

class Tower(object):

    """
    class Tower
    Tower class represent an individual transmission tower.
    """
    def __init__(self, fid, ttype, funct, line_route, design_speed, 
        design_span, terrain_cat, strong_axis, dev_angle, adj=None):

        self.fid = fid # integer
        self.ttype = ttype # Lattice Tower or Steel Pole
        self.funct = funct # e.g., suspension, terminal, strainer
        self.line_route = line_route # string
        self.no_curcuit = 2 # double circuit (default value)
        self.design_speed = design_speed # design wind speed
        self.design_span = design_span # design wind span
        self.terrain_cat = terrain_cat # Terrain Cateogry
        self.strong_axis = strong_axis # azimuth of strong axis relative to North (deg)
        self.dev_angle = dev_angle # deviation angle

        # to be assigned
        self.actual_span = None # actual wind span on eith side
        self.adj = adj #(left, right)
        self.adj_list = None # (23,24,0,25,26) ~ idfy_adj_list (function specific)
        self.adj_design_speed = None
        self.max_no_adj_towers = None # 
        self.cond_pc_adj = None # dict ~ cal_cond_pc_adj
        self.cond_pc_adj_mc = {'rel_idx': None, 'cum_prob': None} # ~ cal_cond_pc_adj

    def calc_adj_collapse_wind_speed(self, terrain_height):
        """
        calculate adjusted collapse wind speed for a tower
        Vc = Vd(h=z)/sqrt(u)*Mz,cat(h=10)/Mz,cat(h=z)
        where u = 1-k(1-Sw/Sd) 
        Sw: actual wind span 
        Sd: design wind span (defined by line route)
        tc: terrain category (defined by line route)
        k: 0.33 for a single, 0.5 for double circuit
        """

        tc_str = 'tc' + str(self.terrain_cat) # Terrain 

        # k: 0.33 for a single, 0.5 for double circuit
        k_factor = {1: 0.33, 2: 0.5} 

        #z = 15.4 for suspension, 12.2 for tension, 12.2 for terminal
        z_dic = {'Suspension': 15.4, 'Strainer': 12.2, 'Terminal': 12.2}

        try: 
            z = z_dic[self.funct]
        except KeyError:
            print "function type of FID %s is not valid: %s" %(self.fid, self.funct)    

        try: 
            mzcat_z = np.interp(z, terrain_height['height'], terrain_height[tc_str])
        except KeyError:
            print "%s is not defined" %tc_str

        mzcat_10 = terrain_height[tc_str][terrain_height['height'].index(10)]

        # calculate utilization factor
        try:
            #print "%s, %s, %s" %(self.fid, self.sd, self.sw)
            u = min(1.0, 1.0 - k_factor[self.no_curcuit]*
            (1.0 - self.actual_span/self.design_span)) # 1 in case sw/sd > 1
        except KeyError:
            print "no. of curcuit %s is not valid: %s" %(self.fid, self.no_curcuit)    

        # Mz,cat(h=10m)
        vc = self.design_speed/np.sqrt(u)*mzcat_10/mzcat_z    

        self.adj_design_speed = vc

        return


    def idfy_adj_list_v2(self, fid2name, cond_pc, flag_strainer=None):
        """
        identify list of adjacent towers which can influence on collapse
        """

        def create_list_idx(idx, nsteps, flag):
            """
                create list of adjacent towers in each direction (flag=+/-1)
            """
            
            list_idx = []
            for i in range(nsteps):
                try:
                    idx = tower[fid2name[idx]].adj[flag]
                except KeyError:
                    idx = -1
                list_idx.append(idx)
            return list_idx

        def mod_list_idx(list_):
            """
            replace id of strain tower with -1
            """
            for i, item in enumerate(list_):

                if item != -1:
                    tf = False
                    try:
                        tf = tower[fid2name[item]].funct in flag_strainer
                    except KeyError:
                        print "KeyError %s" %fid2name[item]

                    if tf == True:
                        list_[i] = -1
            return list_

        try:
            max_no_adj_towers = cond_pc[self.funct]['max_adj']
        except KeyError:
            max_no_adj_towers = cond_pc['Suspension']['max_adj']

        list_left = create_list_idx(self.fid, max_no_adj_towers, 0)
        list_right = create_list_idx(self.fid, max_no_adj_towers, 1)

        #if flag_strainer == None:
        self.adj_list = list_left[::-1] + [self.fid] + list_right
        #else:
        #    self.adj_list = (mod_list_idx(list_left)[::-1] + [self.fid] +
        #                     mod_list_idx(list_right))

        return    

    def idfy_adj_list(self, tower, fid2name, cond_pc, flag_strainer=None):
        """
        identify list of adjacent towers which can influence on collapse
        """

        def create_list_idx(idx, nsteps, flag):
            """
                create list of adjacent towers in each direction (flag=+/-1)
            """
            
            list_idx = []
            for i in range(nsteps):
                try:
                    idx = tower[fid2name[idx]].adj[flag]
                except KeyError:
                    idx = -1
                list_idx.append(idx)
            return list_idx

        def mod_list_idx(list_):
            """
            replace id of strain tower with -1
            """
            for i, item in enumerate(list_):

                if item != -1:
                    tf = False
                    try:
                        tf = tower[fid2name[item]].funct in flag_strainer
                    except KeyError:
                        print "KeyError %s" %fid2name[item]

                    if tf == True:
                        list_[i] = -1
            return list_

        if self.funct in cond_pc.keys():
            self.max_no_adj_towers = cond_pc[self.funct]['max_adj']
        else:
            self.max_no_adj_towers = cond_pc['Suspension']['max_adj']

        list_left = create_list_idx(self.fid, self.max_no_adj_towers, 0)
        list_right = create_list_idx(self.fid, self.max_no_adj_towers, 1)

        if flag_strainer == None:
            self.adj_list = list_left[::-1] + [self.fid] + list_right
        else:
            self.adj_list = (mod_list_idx(list_left)[::-1] + [self.fid] +
                             mod_list_idx(list_right))

        return

    def cal_cond_pc_adj(self, cond_pc, fid2name):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """

        if self.funct in cond_pc.keys():
            cond_pc_adj_mc = copy.deepcopy(cond_pc[self.funct]['prob'])
        else:
            cond_pc_adj_mc = copy.deepcopy(cond_pc['Suspension']['prob'])

        for (i, fid) in enumerate(self.adj_list):
            j = i - self.max_no_adj_towers # convert np index to relative index 
            if fid == -1:
                for key_ in cond_pc_adj_mc.keys():
                    #try: 
                    if j in key_:
                       cond_pc_adj_mc.pop(key_, None)
                    #except TypeError:
                    #    print self.fid, key_

        # sort by cond. prob                
        rel_idx, prob = [], []
        for w in sorted(cond_pc_adj_mc, key=cond_pc_adj_mc.get):
            rel_idx.append(w)
            prob.append(cond_pc_adj_mc[w])

        cum_prob = np.cumsum(np.array(prob))

        self.cond_pc_adj_mc['rel_idx'] = rel_idx
        self.cond_pc_adj_mc['cum_prob'] = cum_prob
        
        # sum by node
        cond_pc_adj = {}
        for key_ in cond_pc_adj_mc.keys():
            for i in key_:
                try:
                    cond_pc_adj[i] += cond_pc_adj_mc[key_]
                except KeyError:
                    cond_pc_adj[i] = cond_pc_adj_mc[key_]

        if cond_pc_adj.has_key(0) == True:
            cond_pc_adj.pop(0)
        
        self.cond_pc_adj = cond_pc_adj

        return


    def cal_cond_pc_adj_v2(self, cond_pc, fid2name):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """
        try:
            cond_pc_adj_mc = copy.deepcopy(cond_pc[self.funct]['prob'])
            max_no_adj_towers = cond_pc[self.funct]['max_adj']
        except KeyError:
            cond_pc_adj_mc = copy.deepcopy(cond_pc['Suspension']['prob'])
            max_no_adj_towers = cond_pc['Suspension']['max_adj']

        for (i, fid) in enumerate(self.adj_list):
            j = i - max_no_adj_towers # convert np index to relative index 
            if fid == -1:
                for key_ in cond_pc_adj_mc.keys():
                    #try: 
                    if j in key_:
                       cond_pc_adj_mc.pop(key_, None)
                    #except TypeError:
                    #    print self.fid, key_

        # sort by cond. prob                
        rel_idx, prob = [], []
        for w in sorted(cond_pc_adj_mc, key=cond_pc_adj_mc.get):
            rel_idx.append(w)
            prob.append(cond_pc_adj_mc[w])

        cum_prob = np.cumsum(np.array(prob))

        self.cond_pc_adj_mc['rel_idx'] = rel_idx
        self.cond_pc_adj_mc['cum_prob'] = cum_prob
        
        # sum by node
        cond_pc_adj = {}
        for key_ in cond_pc_adj_mc.keys():
            for i in key_:
                try:
                    cond_pc_adj[i] += cond_pc_adj_mc[key_]
                except KeyError:
                    cond_pc_adj[i] = cond_pc_adj_mc[key_]

        if cond_pc_adj.has_key(0) == True:
            cond_pc_adj.pop(0)
        
        self.cond_pc_adj = cond_pc_adj

        return
