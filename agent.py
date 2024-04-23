from meteor import Game
import time,threading,sys
        
class MidAgent:
    """Agent part for heavy work."""
    def __init__(self,top):
        self.top = top
    
        # get information for TopAgent
    def ch_m(self,nc,c,i):                  # check for obstacle/meteor
        """Checks for meteors."""
        if nc.ch("obstacle"):                       # obstacle
            return False
        elif not nc.ch("meteore"):                  # no meteor
            return True
        gr = -1
        for x,y,r in self.top.d_dat['meteore']:     # check all meteors
            if (x,y) != nc.p:
                continue
            elif (gr < 0) or gr > r:                # soonest one
                gr = r
        if r <= i:                                  # will be obstacle
            return False
        return True                                 # still clear
    def hexplore(self,p,l_g,lim=-1):        # check cells surrounding 'p'
        """Explores around a given coordinate."""
        d_nodes = {}
        for nc,c,i in self.top.game.hexplore(p,self.ch_m,lim):
            d_nodes[nc.p] = {'p':c.p,
                           'c':[],
                           'i':i}
            if c in d_nodes:
                d_nodes[c.p]['c'].append(nc.p)
            for gp,ch_end in l_g:
                if gp == nc.p and gp != p:
                    yield (d_nodes,gp,ch_end)
    def unroll(self,d_nodes,gp,p):          # get path from 'p' to 'gp'
        """Returns a path from 'p' to 'gp' using 'd_tree'."""
        l_path = []
        if not gp in d_nodes:                           # should not happen
            return l_path
        np = gp
        while np != p:
            np = d_nodes[gp]['p']                       # get parent
            l_path.insert(0,(gp[0]-np[0],gp[1]-np[1]))  # get direction
            gp = np
        return l_path
    def get_cost(self,sp,gp,d_gp):          ## TODO: weigh path cost
        """Function to weigh the cost of a path.
        'sp':   (tuple<int,int>) starting coordinates.
        'gp':   (tuple<int,int>) goal coordinates.
        'd_gp': (dict) info about the goal."""
            # parameter data
        cost = d_gp['cost'] # (int) length of the path by default
        path = d_gp['path'] # list of directions from 'p' to 'gp'
            # game data
        d_dat = self.top.d_dat # game data
        grid_size = d_dat['grid'] # grid size as tuple (width,height)
        list_obstacles = d_dat['obstacle'] # list of obstacle coordinates
        
        ## TODO: weigh the cost by probabilities
        return cost
    def get_tree(self,lim=-1):              # get dict' of paths
        """Returns paths from player+all_people to all_people+all_shelters."""
            # setup
        d_tree,l_end = {self.top.d_dat['joueur']:{}},[]
        for p in self.top.d_dat['personne']:
            d_tree[p] = {}
            l_end.append((p,False))
        for a in self.top.d_dat['abri']:
            l_end.append((a,True))
        for sp in d_tree:                           # iterate
            for d_nodes,gp,ch_end in self.hexplore(sp,l_end,lim):
                l_path = self.unroll(d_nodes,gp,sp)
                d_tree[sp][gp] = {'end':ch_end,
                                  'path':l_path.copy(),
                                  'cost':len(l_path)}
                d_tree[sp][gp]['cost'] = self.get_cost(sp,gp,d_tree[sp][gp])
        return d_tree
        # act on the game
    def start(self):
        """Starts the game."""
        return self.top.game.start()
    def move(self,p,gp):
        """Move until goal or meteor."""
        d_p = self.top.d_tree[p]
        l_path = d_p[gp]['path']
        print("\t",l_path) ##TRACK
        for p in l_path:
            om = len(self.top.d_dat['meteore'])
            self.top.d_dat = self.top.game.move(p)
            time.sleep(1.)
            if ((len(self.top.d_dat['meteore']) != om) or
                self.top.d_dat['end']):
                break
        return False
    def suicide(self):
        """Move up until death."""
        while not self.top.d_dat['end']:
            self.top.d_dat = self.top.game.move((0,-1))
class TopAgent:
    """Agent part for decision-making."""
    def __init__(self,game=None,lim=10):
        self.game = game if game else Game()
        self.mid = MidAgent(self)
        self.lim = lim
        self.d_dat = {}                             # game data
        self.d_tree = {}                            # search tree
        self.l_goal = []                            # objectives in order
    
        # decisions
    def get_tree(self):
        """Get paths from player/people to people/shelters."""
        self.d_tree = self.mid.get_tree(self.lim)
    def utility(self,s1,r1,s2,r2):
        """Check if 'l_min' is better than current."""
        if ((s2 > s1) or ((s2 == s1) and (r2 < r1))):
            return True
        return False
    def decide(self):
        """Get/update the list of objectives."""
        if not self.d_tree:                             # need "tree"
            self.get_tree()
        self.l_goal = []
        l_depth = [(self.d_dat['joueur'],0,0)]
        g_score,g_risk = -1,-1
        while True:
            p,di,r = l_depth[-1]; d_t = self.d_tree[p]  # unpack
            l_k = list(d_t)                             # list of keys
            if di >= len(l_k):                          # backtrack
                l_depth.pop()
                if not l_depth:                         # exhausted
                    break
                l_depth[-1] = (l_depth[-1][0],l_depth[-1][1]+1,
                               l_depth[-1][2])
                continue
            cp = l_k[di]                                # new goal (coords')
            ch_circuit = False                          # check circuits
            for tpl in l_depth:
                if tpl[0] == cp:
                    l_depth[-1] = (p,di+1,r); ch_circuit = True; break
            if ch_circuit:
                continue
            d_p = self.d_tree[p][cp]                    # infos
            risk = r+d_p['cost']
            if not d_p['end']:                          # depth-first
                l_depth.append((cp,0,risk)); continue
            nb_saved = len(l_depth)-3
            nb_saved = 0 if nb_saved < 0 else nb_saved
            score = self.game.get_score(cp,nb_saved)
            if self.utility(g_score,g_risk,score,risk): # better path
                g_score,g_risk = score,r
                self.l_goal = []
                for di2 in range(1,len(l_depth)):
                    self.l_goal.append(l_depth[di2][0])
                self.l_goal.append(cp)
            l_depth[-1] = (p,di+1,r)
        # actions
    def start(self):
        """Starts the game."""
        self.d_dat = self.mid.start()
    def move(self):
        """Tell MidAgent to move."""
        if not self.l_goal:
            return True
        return self.mid.move(self.d_dat['joueur'],self.l_goal[0])
    def loop(self):
        """Game loop but for the agent."""
        if not self.d_dat:                          # auto-start
            self.start()
        while not self.d_dat['end']:                # while not end...
            self.move(); self.get_tree(); self.decide()
            print("Move end:",self.d_dat['joueur'],
                  self.l_goal,self.d_dat['end']) ##TRACK
            if self.d_dat['end']:
                break
            elif (not self.l_goal):                 # got trapped
                print("Time to suicide.")
                self.mid.suicide()
        print(self.d_dat['score'])
        return self.d_dat['score']

if __name__ == "__main__":
    game = Game()               # game intance (with the GUI)
    agent = TopAgent(game)
    def thread_it():
        """A convoluted way to run the GUI and the agent at the same time."""
        thr = threading.Thread(target=agent.loop) # run loop
        thr.start()
    game.d.w.after(1000,thread_it)
    game.d.w.mainloop()
    