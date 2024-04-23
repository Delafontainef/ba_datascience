import grid
import os,re,random,time,json,math

class Meteor:
    """Meteor simulation."""
    def __init__(self,grid,n,ch_o=True):
        self.r = random.randint(4,6)            # remaining time
        self.s = self.set_size(grid,n)          # meteor size
        self.p = self.set_pos(grid,ch_o)        # meteor position
        # initialization
    def set_size(self,grid,n):
        """How big the meteor will be."""
        s = grid.w if grid.w >= grid.h else grid.h
        ns = (2*n)-1 if (2*n)-1 >= 0 else 0
        d = int(s/(10*ns)) if (ns) != 0 else 0
        return (d*n)+1
    def set_pos(self,grid,ch_o=True):
        """Where the meteor will fall."""
        w,h = random.randint(0,grid.w-1),random.randint(0,grid.h-1)
        pg = (random.randint(0,grid.w-1),random.randint(0,grid.h-1))
        c = grid.get(pg)
        if not ch_o or c.d != "obstacle": # obstacles not checked
            return pg
        for c in grid.expand(c):
            if c.d != "obstacle":
                return c.p

class Game:
    """The 'game engine'..."""
    def __init__(self,f="test.json",d=None,p=(0,0)):
        self.f = f
        self.d = d if d else grid.Draw(f=self.f)# Draw class (GUI)
        self.p = p                      # player coordinates
        self.l_meteor = []              # meteor instances
        self.n,self.danger = 0,0        # next meteor, danger status
        self.d_dat = {}                 # data for the agent
        self.unbind()
        # GUI
    def bind(self):
        self.d.w.unbind("<Control-n>")
        self.d.w.bind("<Up>",self.mU)
        self.d.w.bind("<w>",self.mU)
        self.d.w.bind("<Down>",self.mD)
        self.d.w.bind("<s>",self.mD)
        self.d.w.bind("<Left>",self.mL)
        self.d.w.bind("<a>",self.mL)
        self.d.w.bind("<Right>",self.mR)
        self.d.w.bind("<d>",self.mR)
    def unbind(self):
        self.d.w.bind("<Control-n>",self.start)
        self.d.w.unbind("<Up>")
        self.d.w.unbind("<w>")
        self.d.w.unbind("<Down>")
        self.d.w.unbind("<s>")
        self.d.w.unbind("<Left>")
        self.d.w.unbind("<a>")
        self.d.w.unbind("<Right>")
        self.d.w.unbind("<d>")
        # calls
    def hexplore(self,p,f=None,lim=-1):
        """Access to Grid.hexplore method."""
        return self.d.gr.hexplore(self.d.gr.get(p),f,lim)
    def distance(self,p1,p2):
        return self.d.gr.distance(p1,p2)
        # movement
    def _u(self,c,d=""):
        """Updates the cell for 'Draw'."""
        c.d = d; self.d.cell_refresh(c)
    def _resolve(self,c,nc):
        """Moves the player."""
        if (not c) or (not nc) or nc.ch("obstacle"):    # can't move
            return False
        l_p = []                                        # company...
        if not c.ch("joueur"):
            print(c.p) ##DEBUG
        c.i.remove("joueur")                            # restore old value
        c.d = ""
        for a in range(len(c.i)-1,-1,-1):
            if c.i[a] == "joueur":                      # player...
                c.i.pop(a)
            elif c.i[a] == "personne":                  # follow the player...
                l_p.append("personne"); c.i.pop(a)
        if c.p in self.d_dat['personne']:
            self.d_dat['personne'].remove(c.p)
        if c.ch("abri"):                                # color
            c.d = "abri"
        elif c.ch("meteore"):
            c.d = "meteore"
        self._u(c,c.d)                                  # set old cell
        nc.i.append("joueur")                           # move player
        if nc.ch("personne"):
            self.d_dat['personne'].remove(nc.p)
        for p in l_p:                                   # move people
            nc.i.append("personne")
        self._u(nc,"joueur")                            # set new cell
        self.d_dat['joueur'] = self.p = nc.p
        return True
    def move(self,p):
        """Move"""
        c = self.d.gr.get(self.p)
        nc = self.d.gr.get((self.p[0]+p[0],self.p[1]+p[1]))
        self._resolve(c,nc)
        return self.next()
    def mU(self,e=None):
        """Move up."""
        c = self.d.gr.get(self.p)
        nc = self.d.gr.get((self.p[0],self.p[1]-1))
        self._resolve(c,nc)
        return self.next()
    def mD(self,e=None):
        """Move down."""
        c = self.d.gr.get(self.p)
        nc = self.d.gr.get((self.p[0],self.p[1]+1))
        self._resolve(c,nc)
        return self.next()
    def mL(self,e=None):
        """Move left."""
        c = self.d.gr.get(self.p)
        nc = self.d.gr.get((self.p[0]-1,self.p[1]))
        self._resolve(c,nc)
        return self.next()
    def mR(self,e=None):
        """Move right."""
        c = self.d.gr.get(self.p)
        nc = self.d.gr.get((self.p[0]+1,self.p[1]))
        self._resolve(c,nc)
        return self.next()
        # other
    def set_dat(self): # legacy
        """Set data dict' according to inventory (during game)."""
        d_dat = {'score':0,
                 'end':False,
                 'grid':(self.d.gr.w,self.d.gr.h),
                 'futur_meteore':0,
                 'danger':0,
                 'joueur':(0,0),
                 'obstacle':[],
                 'abri':[],
                 'personne':[],
                 'meteore':[]}              # dictionary
        for c in self.d.gr:                 # for each cell...
            for i in c.i:                   # for its inventory...
                if i == "joueur":
                    d_dat[c.d] == c.p
                elif c.d in d_dat:
                    d_dat[c.d].append(c.p)
        return d_dat
    def rand_set(self):
        """Returns a random location that's an empty cell (by color)."""
        p = (random.randint(0,self.d.gr.w-1),
             random.randint(0,self.d.gr.h-1))
        for c in self.d.gr.expand(self.d.gr.get(p)):
            if c.d == "":
                return c
    def draw_meteor(self,m):
        """For a new meteor, draws it on the grid."""
        for c in self.d.gr.expand(self.d.gr.get(m.p),lim=m.s):
            if not c.ch("meteore"):
                c.i.append("meteore")
            self.d_dat['meteore'].append((c.p[0],c.p[1],m.r))
            if not c.ch("joueur"):
                self._u(c,"meteore")
    def check_meteors(self):
        """Update the meteor counter and make 'em fall."""
        for a in range(len(self.l_meteor)-1,-1,-1):
            m = self.l_meteor[a]
            m.r = m.r-1
            if m.r <= 0:        # time to fall
                for c in self.d.gr.expand(self.d.gr.get(m.p),lim=m.s):
                    while "meteore" in c.i:
                        c.i.remove("meteore")
                    if (c.p[0],c.p[1],m.r+1) in self.d_dat['meteore']:
                        self.d_dat['meteore'].remove((c.p[0],c.p[1],m.r+1))
                    if not "obstacle" in c.i:
                        c.i.append("obstacle")
                        self.d_dat['obstacle'].append(c.p)
                    if c.ch("abri"):
                        c.remove("abri")
                    if c.p in self.d_dat['abri']:
                        self.d_dat['abri'].remove(c.p)
                    if c.ch("personne"):
                        c.remove("personne")
                    if c.p in self.d_dat['personne']:
                        self.d_dat['personne'].remove(c.p)
                    self._u(c,"obstacle")
                self.l_meteor.pop(a)
            else:               # update agent data
                for c in self.d.gr.expand(self.d.gr.get(m.p),lim=m.s):
                    i = self.d_dat['meteore'].index((c.p[0],c.p[1],m.r+1))
                    self.d_dat['meteore'][i] = (c.p[0],c.p[1],m.r)
    def get_score(self,c=None,ip=-1):
        c = self.d.gr.get(c) if not isinstance(c,grid.Cell) else c
        if not c.ch("abri"):
            return 0
        if ip < 0:
            ip = 0
            for i in c.i:
                if i == "personne":
                    ip += 1
        return 40 if ip == 0 else 100+((ip-1)*30)
        # states
    def start(self,e=None,p=None):
        """Sets the game up."""
        self.d_dat = {'score':0,'end':False,'joueur':(0,0),
                      'grid':(self.d.gr.w,self.d.gr.h),
                      'futur_meteore':0,'danger':0,
                      'obstacle':[],'abri':[],'personne':[],
                      'meteore':[]}             # player data
        self.d.load(self.f)
        self.d.draw()
        ch_people,ch_shelter = False,False
        s = self.d.gr.w if self.d.gr.w >= self.d.gr.h else self.d.gr.h
        sm = int(s/4)
        for c in self.d.gr:                     # set grid inventory...
            c.i = []
            if not c.d:
                continue
            c.i = [c.d] if c.d else []          # based on color...
            if c.d == "abri":
                ch_shelter = True
                self.d_dat['abri'].append(c.p)
            elif c.d == "joueur":
                if not p:
                    self.d_dat['joueur'] = self.p = p = c.p
            elif c.d == "personne":
                ch_people = True; self.d_dat['personne'].append(c.p)
            elif c.d in self.d_dat:
                self.d_dat[c.d].append(c.p)
        if not p:                               # player coordinates
            self.p = self.rand_set().p
            self.p = (random.randint(0,self.d.gr.w-1),
                      random.randint(0,self.d.gr.h-1))
            for c in self.d.gr.expand(self.d.gr.get(self.p)):
                if c.d == "":
                    self.p = c.p
                    c.add("joueur")
                    self._u(c,"joueur"); break
            self.d_dat['joueur'] = self.p
        if not ch_people:                       # people to save (random)
            for a in range(random.randint(s-sm,s+sm)):
                c = self.rand_set()
                c.add("personne"); self._u(c,"personne")
                self.d_dat['personne'].append(c.p)
        if not ch_shelter:                      # shelters (random)
            for a in range(int(s/3)):
                c = self.rand_set()
                c.add("abri"); self._u(c,"abri")
                self.d_dat['abri'].append(c.p)
        self.n = random.randint(3,5)            # meteor countdown
        self.bind()                             # allow movement
        return self.d_dat
    def next(self):
        """Advances the game (handles meteors...)."""
        self.check_meteors()
        c = self.d.gr.get(self.p)
        if c.ch("abri") or c.ch("obstacle"):    # end game
            self._u(c,"abri")
            return self.end()
        self.n = self.n-1                       # meteor countdown
        if self.n <= 0:
            self.n = random.randint(3,5)
            self.danger += 1
            if self.danger > 5 and self.danger < 10:
                self.danger = 10
            for a in range(self.danger):        # generate meteors
                for b in range(self.danger-a):
                    self.l_meteor.append(Meteor(self.d.gr,a))
                    self.draw_meteor(self.l_meteor[-1])
        self.d_dat['futur_meteore'] = self.n
        self.d_dat['danger'] = self.danger
        return self.d_dat
    def end(self):
        """Ends the game."""
        self.unbind()                           # block movement
        self.d_dat['end'] = True
        c = self.d.gr.get(self.p)
        self.d_dat['score'] = s = self.get_score(c)
        c.remove("joueur")
        self.p = (0,0)
        self.n,self.danger = 0,0
        self.l_meteor.clear()
        return self.d_dat
if __name__ == "__main__":
    g = Game("20.json")
    g.d.w.mainloop()