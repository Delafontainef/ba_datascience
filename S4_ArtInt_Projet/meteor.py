import grid
import os,re,time,json,math

class Game:
    """The 'game engine'..."""
    def __init__(self,d=None,p=(0,0)):
        self.d = d  # Draw class (GUI)
        self.p = p
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
        # methods
    def manhattan(self,x,y):
        """Spreads cost from 'x/y' in a manhattan way (for Dijkstra)."""
        pass
        # movement
    def _u(self,c,d=""):
        """Updates the cell for 'Draw'."""
        c.d = d; self.d.cell_refresh(c)
    def _resolve(self,c,nc):
        """Moves the player."""
        if (not c) or (not nc) or nc.ch("obstacle"):    # can't move
            return False
        l_p = []                                        # company...
        c.i.remove("joueur")                            # restore old value
        c.d = ""
        for a in range(len(c.i)-1,-1,-1):
            if c.i[a] == "joueur":                      # player...
                c.i.pop(a)
            elif c.i[a] == "personne":                  # follow the player...
                l_p.append("personne"); c.i.pop(a)
            elif c.i[a] == "abri":
                c.d = "abri"
        self._u(c,c.d)
        nc.i.append("joueur")
        for p in l_p:
            nc.i.append("personne")
        self._u(nc,"joueur")
        return True
    def mU(self,e=None):
        """Move up."""
        c = self.d.gr.get(self.p[0],self.p[1])
        nc = self.d.gr.get(self.p[0],self.p[1]-1)
        if self._resolve(c,nc):
            self.p = (self.p[0],self.p[1]-1)
        self.next()
    def mD(self,e=None):
        """Move down."""
        c = self.d.gr.get(self.p[0],self.p[1])
        nc = self.d.gr.get(self.p[0],self.p[1]+1)
        if self._resolve(c,nc):
            self.p = (self.p[0],self.p[1]+1)
        self.next()
    def mL(self,e=None):
        """Move left."""
        c = self.d.gr.get(self.p[0],self.p[1])
        nc = self.d.gr.get(self.p[0]-1,self.p[1])
        if self._resolve(c,nc):
            self.p = (self.p[0]-1,self.p[1])
        self.next()
    def mR(self,e=None):
        """Move right."""
        c = self.d.gr.get(self.p[0],self.p[1])
        nc = self.d.gr.get(self.p[0]+1,self.p[1])
        if self._resolve(c,nc):
            self.p = (self.p[0]+1,self.p[1])
        self.next()
        # states
    def start(self,e=None,p=None):
        """Sets the game up."""
        self.p = p if p else self.p
        for c in self.d.gr:                     # set grid inventory...
            if c.p == self.p:                   # player location!
                c.i = ["joueur"]
                self._u(c,"joueur")
            elif not c.d:
                continue
            c.i = [c.d] if c.d else []          # based on color...
            if c.d == "abri":
                c.c = 100                       # eh.
        self.bind()                             # allow movement
    def next(self):
        """Advances the game (handles meteors...)."""
        c = self.d.gr.get(self.p[0],self.p[1])
        if c.ch("abri") or c.ch("obstacle"):    # end game
            self._u(c,"abri")
            self.end()
    def end(self):
        """Ends the game."""
        self.unbind()                           # block movement
        c = self.d.gr.get(self.p[0],self.p[1])
        if not c.ch("abri"):
            print(f"Score: 0"); return
        s = 100
        ip = 0
        for i in c.i:
            if i == "personne":
                ip += 1
        if ip == 0:
            s = 40
        else:
            s = s+(ip*30)
        print(f"Score: {s}")
    
if __name__ == "__main__":
    g = Game(grid.Draw(f="test.json"))
    g.d.w.mainloop()