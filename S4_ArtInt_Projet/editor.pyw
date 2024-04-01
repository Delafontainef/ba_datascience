import tkinter as tk
from tkinter import ttk,filedialog
import os,re,time,json,math
import grid

class New:
    """New grid popup."""
    def __init__(self,parent,w):
        self.par = parent
        self._build_gui(w)
        
    def _build_gui(self,w):
        """GUI elements for the popup."""
        self.s = tk.Toplevel(w)                                 # popup
        self.x = tk.StringVar(self.s,value="1")
        self.y = tk.StringVar(self.s,value="1")
        self.s.title("Nouvelle grille")
        vld = (self.s.register(self._ch_entry),'%P')            # fields
        e1 = ttk.Entry(self.s,textvariable=self.x,validate="key",
                       validatecommand=vld)
        e2 = ttk.Entry(self.s,textvariable=self.y,validate="key",
                       validatecommand=vld)
        b = ttk.Button(self.s,text="Créer",command=self._ret)  # button
        e1.grid(row=0,column=0); e2.grid(row=0,column=1)
        b.grid(row=1,column=0,columnspan=2)
    def _ch_entry(self,val):
        """Validates entry variables."""
        if not val:                         # allow empty val
            return True
        elif not re.match("[0-9]+$",val):   # but only integers
            return False
        val = int(val)                      # between ]0,100]
        return True if (val > 0 and val <= 100) else False
    def _ret(self):
        """Tells the editor the final value."""
        x = self.x.get(); y = self.y.get()
        x = "1" if not x else x
        y = "1" if not y else y
        self.s.destroy()
        self.par.new(int(x),int(y))
class Editor:
    """Handles map drawing."""
    def __init__(self,w,d=None,f=""):
        self.w = tk.Tk() if not w else w
        self.d = grid.Draw(self.w,f) if not d else d
        self.f = f
        self.l_col = self.list_col()
        self.build_gui()
        self.bind()
    
    def list_col(self):
        """Turns the color dict' into a list."""
        return [k for k in self.d.d_col]
    def load(self,f):
        """Loads a json."""
        self.d.load(f)                  # let 'Draw' handle it
        self.l_col = self.list_col()
        self.d.draw()
    def new(self,x=-1,y=-1):
        """Creates a new grid."""
        if x <= 0 or y <= 0:
            return New(self,self.w)
        self.d.gr = Grid(x,y)
        self.d.draw()
    def load_gui(self):
        """Loads a file via GUI."""
        self.f = filedialog.askopenfilename(filetypes=[("json",".json")],
                                            defaultextension=".json")
        self.load(self.f)
    def save(self,f=None):
        """Saves as json."""
        f = self.f if not isinstance(f,str) else f
        if not f:               # need a file path
            self.f = filedialog.asksaveasfilename(filetypes=[("json",".json")],
                                                  defaultextension=".json")
        self.d.save(self.f)     # let 'Draw' handle it
    def save_as(self,e):
        """Saves as json. Ensures asking for a file location."""
        self.save("")
    def build_gui(self):
        """Add menu."""
        menu = tk.Menu(self.w)
        fm = tk.Menu(menu,tearoff=0)
        fm.add_command(label="Nouveau",command=self.new)
        fm.add_command(label="Ouvrir",command=self.load_gui)
        fm.add_command(label="Sauvegarder",command=self.save)
        fm.add_command(label="Sauvegarder sous...",command=self.save_as)
        fm.add_separator()
        fm.add_command(label="Quitter",command=self.w.quit)
        menu.add_cascade(label="Fichier",menu=fm)
        self.w.config(menu=menu)
    def bind(self):
        """Add mouse controls."""
        self.d.c.bind("<Button-1>",self.next_col)   # left-clic
        self.d.c.bind("<Button-3>",self.prev_col)   # right-clic
        self.w.bind("<Control-n>",self.new)         # Ctrl+n
        self.w.bind("<Control-s>",self.save)        # Ctrl+s
        self.w.bind("<Control-Alt-s>",self.save_as) # Ctrl+alt+s
        self.w.bind("<Control-o>",self.load_gui)    # Ctrl+o
    def prev_col(self,e):
        """Changes the cell's color id (previous)."""
        c = self.d.get_cell(e.x,e.y)
        if not c:
            return
        i = self.l_col.index(c.d)
        i = len(self.l_col)-1 if i-1 < 0 else i-1
        c.d = self.l_col[i]
        self.d.cell_refresh(c)
    def next_col(self,e):
        """Changes the cell's color id (next)."""
        c = self.d.get_cell(e.x,e.y)
        if not c:
            return
        i = self.l_col.index(c.d)
        i = 0 if i+1 >= len(self.l_col) else i+1
        c.d = self.l_col[i]
        self.d.cell_refresh(c)

if __name__ == "__main__":
    w = tk.Tk()
    e = Editor(w,f="test.json")
    w.mainloop()