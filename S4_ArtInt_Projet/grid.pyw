import tkinter as tk
import os,re,time,json,math

GR_N = [(0,1),(1,0),(0,-1),(-1,0)]          # all directions
    # file functions
def load_json(f):
    """Loads a dict from a json file."""
    if isinstance(f,dict):                          # dictionary
        d_dat = f
    elif isinstance(f,str) and os.path.isfile(f):   # path
        with open(f,'r',encoding="utf-8") as rf:    # json it
            d_dat = json.load(rf)
    else:
        return {}
    return d_dat
def save_json(f,d_dat):
    """Saves a dict 'd_dat' into a json file."""
    if f:
        wf = open(f,'w',encoding="utf-8")
        json.dump(d_dat,wf,ensure_ascii=False,indent=4)
        wf.close()

class Cell:
    """A grid cell."""
    def __init__(self,pos=(0,0),cost=0,inv=[],grid=None):
            # general
        self.p = pos                        # position
        self.gr = grid                      # grid
        self.c = cost                       # cost
        self.i = inv                        # inventory
            # tkinter
        self.d = ""                         # drawing
        self.tk = None                      # drawn object
    
    def set_pos(self,x,y):
        """Change the cell's position."""
        x = self.p[0] if not x else x
        y = self.p[1] if not y else y
        self.p = (x,y)
    def add(self,x):
        """Adds to inventory."""
        if x not in self.i:
            self.i.append(x)
    def remove(self,x):
        """Remove from inventory."""
        if x in self.i:
            self.i.remove(x)
    def ch(self,x):
        """Check inventory for 'x'."""
        return True if x in self.i else False
    def next(self,ch_none=False,ch_f=None):
        """Returns adjacent cells.
        'ch_none' to ignore missing cells.
        'ch_f' to pass a function ('nc' as only parameter)."""
        l_res = []
        if not self.gr:                         # no grid
            return l_res
        x,y = self.pos                          # our position
        for tpl in GR_N:                        # for each direction
            nx,ny = x+tpl[0],y+tpl[0]
            nc = self.gr.get(nx,ny)            # ask grid for cell
            if (ch_none and (not nc)) or (ch_f and (not ch_f(nc))):
                continue
            l_res.append(self.gr.get(nx,ny))   # add cell to result
        return l_res
class Grid:
    """The grid."""
    def __init__(self,x=0,y=0,f=None):
        self.w,self.h = y,x
        self.grid = self.gen_grid(x,y)
        if f:
            self.load(f)
        # generic
    def __bool__(self): # always true
        return True
    def __iter__(self): # iterates over cells
        for line in self.grid:
            for cell in line:
                yield cell
    def __len__(self):  # number of cells
        lc = 0
        for c in self:
            if c:       # allow empty slots?
                lc+=1
        return lc
        # operations
    def gen_grid(self,x,y):
        """Generates a matrix of cells."""
        self.w,self.h = y,x
        return [[Cell((i,j),grid=self) for i in range(x)] for j in range(y)]
    def get(self,x,y):
        """Returns Cell at coordinates (x,y)."""
        if (not self.grid) or (not self.grid[0]):           # no grid
            return None
        if (x < 0 or x >= self.w) or (y < 0 or y >= self.h):# out of range
            return None
        return self.grid[y][x]
        # files
    def load(self,f):
        """Loads a grid from json."""
        d_grid = load_json(f)
        d_grid = d_grid['grid'] if 'grid' in d_grid else d_grid
        self.grid = self.gen_grid(*d_grid['size'])
        for d_cell in d_grid['cells']:
            if (not 'pos' in d_cell):
                continue
            cell = self.get(*d_cell['pos'])
            if not cell:
                continue
            if 'cost' in d_cell:
                cell.c = d_cell['cost']
            if 'inventory' in d_cell:
                cell.i = d_cell['inventory']
            if 'color' in d_cell:
                cell.d = d_cell['color']
    def save(self,f=None):
        """Saves grid in json format."""
        d_grid = {'size':[self.w,self.h],
                  'cells':[]}
        for cell in self:
            if (not cell) or (not (cell.c != 0 or cell.i or cell.d)):
                continue
            d_grid['cells'].append({'pos':cell.p})
            if cell.c != 0:
                d_grid['cells'][-1]['cost'] = cell.c
            if cell.i:
                d_grid['cells'][-1]['inventory'] = cell.i
            if cell.d:
                d_grid['cells'][-1]['color'] = cell.d
        if f:
            save_json({'grid':d_grid})
        return d_grid
class Draw:
    """The GUI."""
    def __init__(self,w=None,f="",dim=(800,800),m=(0,0),ts=17,
                 colors={"":"red"}):
        self.w = w if w else tk.Tk()        # window
        self.gr = Grid(*m)                  # grid
        self.d_col = colors                 # dict of colors...
        self.ts = ts
            # internals
        self.ox,self.oy,self.sz = 0,0,80
        self.v_area = (0,0,88*m[0],88*m[1])
        self.draw_id = None
        self.ch_resize,self.l_refresh = False,[]
            # load stuff
        self.load(f)                        # load graph
        self._build_gui(*dim)               # build interface
        # self.c
        self.draw()                         # first drawing
        self.w.after(self.ts,self.refresh)
        
    def load(self,f):
        """Loads a json."""
        d_draw = load_json(f)
        if 'grid' in d_draw:                # grid data
            self.gr.load(d_draw['grid'])
        if not 'draw' in d_draw:
            return
        d_draw = d_draw['draw']
        if 'colors' in d_draw:              # colors
            self.d_col = d_draw['colors']
    def save(self,f=None):
        """Saves as json."""
        d_draw = {'draw':{'colors':self.d_col},
                  'grid':self.gr.save()}
        if f:
            save_json(f,d_draw)
        return d_draw
    
        # properties set/get
    def get_pad(self):
        """Returns the padding between cells."""
        p = int(self.sz//10)
        return 2 if p < 2 else p
    def get(self,x,y): # get cell from GUI coordinates
        """Returns the cell instance according to coordinates."""
        p = self.get_pad()
        px = int((x+self.v_area[0]-self.ox)//(self.sz+p))
        py = int((y+self.v_area[1]-self.oy)//(self.sz+p))
        return self.gr.get(px,py)
    def cell_refresh(self,c):
        """Asks 'Draw' to redraw that cell."""
        self.l_refresh.append(c)
    def set_refresh_rate(self,ts):
        """Changes the refresh rate."""
        if isinstance(ts,float):            # assumed seconds
            self.ts = int(ts*1000)
        else:                               # assumed milliseconds
            self.ts = ts
        # dimensions and resize
    def get_v_area(self):
        """Returns the visible area coordinates."""
        return (self.c.canvasx(0),self.c.canvasy(0),
                self.c.canvasx(self.c.winfo_width()),
                self.c.canvasy(self.c.winfo_height()))
    def get_v_pos(self,p=None):
        """Returns the start/end grid positions for the visible area."""
        p = self.get_pad() if not p else p
        mxy = int((self.sz+p)*self.gr.w)+p  # matrix size (x/y axes)
        px = 0 if self.v_area[0] <= self.ox else \
             int((self.v_area[0]-self.ox)//(self.sz+p))
        py = 0 if self.v_area[1] <= self.oy else \
             int((self.v_area[1]-self.oy)//(self.sz+p))
        pnx = self.gr.w if self.v_area[2] >= self.ox+mxy else \
             int((self.v_area[2]-self.ox)//(self.sz+p))+1
        pny = self.gr.h if self.v_area[3] >= self.oy+mxy else \
             int((self.v_area[3]-self.oy)//(self.sz+p))+1
        pnx = self.gr.w if pnx > self.gr.w else pnx
        pny = self.gr.h if pny > self.gr.h else pny
        return px,py,pnx,pny
    def on_scr_h(self,x,y):
        """If the canvas is moved in any way shape or form,
        this will detect it."""
        self.v_area = vx,vy,w,h = self.get_v_area()     # visible area
        w = w-vx if w-vx >= 0 else 0
        h = h-vy if w-vx >= 0 else 0
        wx,hy = self.gr.w,self.gr.h
        dw = int(w//(wx)*0.9) if wx > 0 else 0          # new canvas size
        dh = int(h//(hy)*0.9) if hy > 0 else 0
        ch_w = True if dw < dh else False               # new cell size
        if (ch_w and dw > 80) or (dh > 80):
            self.sz = 80
        elif (ch_w and dw < 10) or (dh < 10):
            self.sz = 10
        else:
            self.sz = dw if ch_w else dh
        p = self.get_pad()                              # padding
        self.ox = int((w-((self.sz+p)*wx)+p)//2)        # offset (x axis)
        self.ox = 0 if self.ox < 0 else self.ox
        self.oy = int((h-((self.sz+p)*hy)+p)//2)        # offset (y axis)
        self.oy = 0 if self.oy < 0 else self.oy
        self.ch_resize = True                           # ask for refresh
        self.scr_h.set(x,y)                             # apply to scrollbar
        # GUI
    def _build_gui(self,x,y):
        """Builds the GUI. Adds 'self.c' to properties."""
            # canvas
        main_frame = tk.Frame(self.w,width=x,height=y)
        self.c = tk.Canvas(main_frame)
        self.c.grid(row=0,column=0,sticky="news")
        self.scr_h = tk.Scrollbar(main_frame,orient=tk.HORIZONTAL,
                              command=self.c.xview)
        self.scr_h.grid(row=1,column=0,sticky="ew")
        self.scr_v = tk.Scrollbar(main_frame,orient=tk.VERTICAL,
                              command=self.c.yview)
        self.scr_v.grid(row=0,column=1,sticky="ns")
        self.c.config(xscrollcommand=self.on_scr_h,
                      yscrollcommand=self.scr_v.set)
        self.c.bind('<Configure>',
               lambda e: self.c.configure(scrollregion=self.c.bbox("all")))
        main_frame.grid(row=0,column=0,sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.columnconfigure(self.w,0,weight=1)
        tk.Grid.rowconfigure(self.w,0,weight=1)
        main_frame.columnconfigure(0,weight=1)
        main_frame.rowconfigure(0,weight=1)
    def _set_cell(self,c,p=None):
        """Finds the cell's new coordinates."""
        if not p:       # ideally padding is already a parameter
            p = self.get_pad()
        x,y = self.ox+(c.p[0]*(self.sz+p)),self.oy+(c.p[1]*(self.sz+p))
        nx,ny = x+self.sz,y+self.sz
        return x,y,nx,ny
    def _create_cell(self,c,p=None):
        """Draws a given cell (for self.draw() only)."""
        x,y,nx,ny = self._set_cell(c,p)
        c.tk = self.c.create_rectangle(x,y,nx,ny,outline="")
        self.c.itemconfigure(c.tk,fill=self.d_col[c.d]) # coloring
    def draw(self):
        """Generates objects for the canvas."""
        if not self.c:                              # no way to draw (canvas)
            return
        self.c.delete("all")                        # clear canvas
        p = self.get_pad()
        for l in self.gr.grid:                      # for each line...                        
            for c in l:                             # for each cell...
                if not c:                           # allow no cell?
                    continue
                self._create_cell(c,p)
    def refresh(self):
        """Refreshes the canvas."""
        p = self.get_pad()
        if self.ch_resize:
            # px,py,pnx,pny = self.get_v_pos(p)
            # self.c.delete("all")                    # clear canvas
            # for h in range(py,pny):                 # only relevant cells
                # for w in range(px,pnx):
                    # self._create_cell(self.gr.get(w,h),p)
            # if px > 0 or py > 0:                    # for scrollregion
                # self._create_cell(self.gr.get(0,0),p)
            # if pnx < self.gr.w or pny < self.gr.h:
                # self._create_cell(self.gr.get(self.gr.w-1,self.gr.h-1),p)
            for cell in self.gr:                    # resize
                x,y,nx,ny = self._set_cell(cell,p)
                self.c.coords(cell.tk,x,y,nx,ny)
            self.ch_resize = False
        else:                                       # change color
            for cell in self.l_refresh:
                self.c.itemconfigure(cell.tk,fill=self.d_col[cell.d])
        self.w.after(self.ts,self.refresh)
    
if __name__ == "__main__":
    w = tk.Tk()
    d = Draw(w,f="test.json")
    w.mainloop()