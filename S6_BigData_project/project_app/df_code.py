import numpy as np
import os, time

def _args(kw):
    """Extracts parameters from kwargs."""
    kw = dict(kw)
    s, c = kw.get('size', 10), kw.get('cycle', 10)
    io = kw.get('io', False)
    s = int(s) if not isinstance(s, int) else s
    c = int(c) if not isinstance(c, int) else c
    if isinstance(io, str):
        io = True if io.lower().startswith('t') else False
    kw['size'] = s; kw['cycle'] = c; kw['io'] = io
    return s, c, io, kw

def _iter_dat(dat, c):
    """Iterates 'c' times over 'dat'."""
    for j in range(c):
        for i, e in enumerate(dat):
            yield i, e

def _io(i, e, k, lim=0.01):
    """Writes value 'e' in 'dat.txt'.
       Erases if opening takes too long."""
    with k:
        if not os.path.isfile('dat.txt'):
            wf = open('dat.txt', 'w', encoding="utf-8")
            wf.write(f"{e}\n"); wf.close(); return
        ch = time.time()
        rf = open("dat.txt", 'r', encoding="utf-8")
        rf.close()
        m = 'w' if time.time()-ch >= lim else 'a'
        with open("dat.txt", m, encoding="utf-8") as wf:
            wf.write(f"{e}\n")

def _loop(dat, c, io, k):
    """Loop 'c' times over the 'dat' list.
       'io' boolean on whether to read/write 'dat.txt'."""
    for i, e in _iter_dat(dat, c):
        dat[i] = e + np.random.randint(-1, 2)
        if io and np.random.randint(0, 101) == 0:
            _io(i, e, k)

def run(kw, k):
    """Wastes CPU/memory resources.
       'kw' is kwargs, a dict."""
    start = time.time()  # time all operations
    s, c, io, kw = _args(kw) # get parameters
    dat = [a for a in range(s)] # make memory work
    if c <= 0.:          # leave CPU alone
        time.sleep(np.random.uniform(0.01, 4.001))
        # time.sleep(0.5)
    else:                # make the CPU work
        _loop(dat, c, io, k)
    kw['time'] = time.time()-start
    return kw		 # return dict
