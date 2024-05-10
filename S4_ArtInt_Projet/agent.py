from meteor import Game
import time,threading,sys
        
class MidAgent:
    """Agent part for heavy work."""
    def __init__(self,top,st=1.):
        self.top = top
        self.st = st
    
        # get information for TopAgent
    def hexplore(self,sp,l_g,ch_a,lim=-1):    # check cells surrounding 'p'
        """Explores around a given coordinate."""
        d_nodes = {}
        for nc,c,l_c,i in self.top.game.hexplore(sp,ch_a,lim):
            d_nodes[nc] = {'p':c,
                           'c':[],
                           'i':i,
                           'nb_mp':{}} # pos,children,nb_steps,probs
            if  "meteore" in l_c:
                for x,y,r,pr in self.top.d_dat['meteore']: # check all meteors
                    if (x,y) != nc:
                        continue
                    elif pr not in d_nodes[nc]['nb_mp']:
                        d_nodes[nc]['nb_mp'][pr] = 1
                    else:
                        d_nodes[nc]['nb_mp'][pr] += 1
            if c in d_nodes:                # parent
                d_nodes[c]['c'].append(nc)
            for gp,ch_end in l_g:           # for each goal...
                if gp == nc and gp != sp:
                    yield (d_nodes,gp,ch_end)
    def unroll(self,d_nodes,gp,p):           # get path from 'p' to 'gp'
        """Returns a path from 'p' to 'gp' using 'd_tree'."""
        l_path = []
        if not gp in d_nodes:                           # should not happen
            return l_path
        np = gp
        while np != p:
            np = d_nodes[gp]['p']                       # get parent
            l_path.insert(0,(gp[0]-np[0],gp[1]-np[1],
                             d_nodes[gp]['nb_mp']))  # get direction
            gp = np
        return l_path
    def rec_cost(self,nb_c,gr,danger,gprob):
        """For recursivity's sake."""
        if danger >= 5 and danger < 10:
            danger = 10
        val = 0
        if nb_c <= 1:
            val= 0
        elif nb_c == 2:
            val= 1-(((gr-1)/gr)**danger)*gprob
        elif nb_c == 3:
            val = 1-(((gr-4.5)/gr)**danger)*gprob
        elif nb_c > 3:
            val = (1-((((gr-(2.5*nb_c)+(9/4))/gr)**danger)*
                    (1-self.rec_cost(nb_c-4,gr,danger+1,gprob))))
        return val
    def get_cost(self,l_path):               ## TODO: weigh path cost
        """Function to weigh the cost of a path.
        'd_gp': (dict) info about the goal."""
            # parameter data
        nb_c = len(l_path)
        if nb_c > 100:                   # safety
            return 1
        T = 4
        d_gprob = {0.25:0,0.5:0,1.:0}    # nb_meteors by probability
        for x,y,d_prob in l_path:
            for k,v in d_prob.items():
                if k in d_gprob:
                    d_gprob[k] += v
            # game data
        d_dat = self.top.d_dat # game data
        grid_size = d_dat['grid'] # grid size as tuple (width,height)
        gr = grid_size[0]*grid_size[1] # taille grille
        danger = d_dat['danger']
        gprob = 0
        if d_gprob[1.] == 0:
            gprob = ((1/2)**(d_gprob[0.5]))*((3/4)**(d_gprob[0.25]))
        return self.rec_cost(nb_c,gr,d_dat['danger'],gprob)
    def get_path(self,d_tree,l_end,ch_a,lim=-1):
        """Use hexplore() on a given set 'l_end'."""
        for sp in d_tree:                           # iterate
            for d_nodes,gp,ch_end in self.hexplore(sp,l_end,ch_a,lim):
                l_path = self.unroll(d_nodes,gp,sp)
                d_tree[sp][gp] = {'end':ch_end,
                                  'path':l_path,
                                  'cost':len(l_path)}
    def get_tree(self,lim=-1):              # get dict' of paths
        """Returns paths from player+all_people to all_people+all_shelters."""
            # setup
        d_tree,l_end = {self.top.d_dat['joueur']:{}},[]
        for p in self.top.d_dat['personne']:
            d_tree[p] = {}
            l_end.append((p,False))
        self.get_path(d_tree,l_end,True,lim) # iterate (people)
        l_end = []
        for a in self.top.d_dat['abri']:
            l_end.append((a,True))
        self.get_path(d_tree,l_end,False,lim) # iterate (shelters)
        return d_tree
        # act on the game
    def start(self):
        """Starts the game."""
        return self.top.game.start()
    def move(self,p,gp):
        """Move until goal or meteor."""
        d_p = self.top.d_tree[p]
        l_path = d_p[gp]['path']
        for tpl in l_path:
            om = len(self.top.d_dat['meteore'])
            self.top.d_dat = self.top.game.move(tpl[0:2])
            time.sleep(self.st)
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
    def __init__(self,game=None,lim=10,st=.4):
        self.game = game if game else Game()
        self.mid = MidAgent(self,st=st)
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
        if (s1 < 0.) or (s2*(1-r2)) > (s1*(1-r1)):
            return True
        return False
    def decide(self):
        """Get/update the list of objectives."""
        if not self.d_tree:                             # need "tree"
            self.get_tree()
        self.l_goal = []
        l_depth = [(self.d_dat['joueur'],0,[])]
        g_score,g_risk = -1,-1
        debug = 1000
        while True:
            debug = debug-1
            if debug <= 0:
                break
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
            r = r+d_p['path']
            risk = self.mid.get_cost(r)
            if not d_p['end']:                          # depth-first
                l_depth.append((cp,0,r)); continue
            nb_saved = len(l_depth)-1
            nb_saved = 0 if nb_saved < 0 else nb_saved
            score = self.game.get_score(cp,nb_saved)
            if self.utility(g_score,g_risk,score,risk): # better path
                g_score,g_risk = score,risk
                self.l_goal = []
                for di2 in range(1,len(l_depth)):
                    self.l_goal.append(l_depth[di2][0])
                self.l_goal.append(cp)
                # print("YUP:",self.l_goal,nb_saved,score,risk) ## TRACK
            else:
                l_tmp = []
                for di2 in range(1,len(l_depth)):
                    l_tmp.append(l_depth[di2][0])
                # print("NOPE:",l_tmp,nb_saved,score,risk) ## TRACK
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
            if self.d_dat['end']:
                break
            elif (not self.l_goal):                 # got trapped
                print("Time to suicide.")
                self.mid.suicide()
            print("Move end:",self.d_dat['joueur'],
                  self.l_goal,self.d_dat['end']) ##TRACK
        print(self.d_dat['score'])
        return self.d_dat['score']

if __name__ == "__main__":
    game = Game("20.json")               # game intance (with the GUI)
    agent = TopAgent(game,lim=100,st=.2)
    def thread_it():
        """A convoluted way to run the GUI and the agent at the same time."""
        thr = threading.Thread(target=agent.loop) # run loop
        thr.start()
    # agent.loop()
    game.d.w.after(3000,thread_it)
    game.d.w.mainloop()
    