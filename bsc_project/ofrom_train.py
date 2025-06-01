""" 29.05.2025
Simulates passive/active training.

Supports command calls:
    python test.py function [args/kwargs]
With supported functions:
- 'plot': Plots the accuracy based on a pre-saved json file.
          ('plot_acc' function)
- 'passive': Tests conventional (passive) training.
             ('save_passive' function)
- 'active': Tests active training.
            The strategy is hardcoded in the 'Gen' class.
            ('save_active' function)
The list of parameters (args/kwargs) is in the '_args' function,
along with their default values.
            
When testing, we:
- increment 'loop' times an initially empty subset 
  by batches of 'lim' tokens
- train it and get an accuracy score
- select the next batch:
|- passive) at random,
|- active)  by file with lowest average confidence score
- we repeat that process 'it' times and save the accuracy each time 
  in file 'f'.

When plotting, we load the saved accuracy and plot its average plus CI.

Note: Anything about scikit-learn should be in 'ofrom_crf.py'.
      Anything about the data itself should be in 'ofrom_gen.py'.
      This script should be purely for looping and plotting.
Note: Attempts at parallel processing have proven slower
      (threads, processes, pool).
"""

import sys, os, re, time, json, joblib
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                'code')) # ...
from ofrom_gen import Gen
from scipy.stats import t as sci_t
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

    # Support #
    #---------#
def prt(txt, verbose=True):
    """Overwrites current line."""
    if verbose:
        print("\r"+" "*100+"\r"+txt, end="")
def _plt(x, y, alpha=0.95, color="b", label="", ax=None):
    """Plots a graph with a CI."""
    my, cuy, cdy = [], [], []
    for el in y:                    # manual build of mean +/- var
        my.append(np.mean(el))
        mod = np.std(el)/np.sqrt(len(el))
        v_el = sci_t.interval(alpha, df=len(el)-1, loc=my[-1],
                              scale=mod)
        cuy.append(v_el[1])
        cdy.append(v_el[0])
    if ax:
        ax.plot(x, my, color=color, linewidth=2, label=label)
        ax.plot(x, cuy, linestyle='--', color=color, linewidth=1)
        ax.plot(x, cdy, linestyle='--', color=color, linewidth=1)
    return x, my, cuy, cdy
def _plt_ci(y, xm, alpha, title, ch_graph=True):
    """Plots CI functions."""
    x = [(i+1)*xm for i in range(len(y))] # /!\ it > 0
    my, cuy, cdy = [], [], []
    for el in y:                    # manual build of mean +/- var
        my.append(np.mean(el))
        mod = np.std(el)/np.sqrt(len(el))
        v_el = sci_t.interval(alpha, df=len(el)-1, loc=my[-1],
                              scale=mod)
        cuy.append(v_el[1])
        cdy.append(v_el[0])
    if not ch_graph:
        return x, my, cuy, cdy
    plt.plot(x, my, color='b', linewidth=2)
    plt.plot(x, cuy, linestyle='--', color='b', linewidth=1)
    plt.plot(x, cdy, linestyle='--', color='b', linewidth=1)
    plt.title(title)
    plt.xlabel("Token count (thousands)")
    plt.ylabel(f"Accuracy score ({len(y[0])})")
    title = title.replace(" ", "_").lower()+".png"
    plt.savefig(title)
    return x, my, cuy, cdy

def _wrap(fun, lv, i, *args):
    """Wraps around a function for multi-threading/processing."""
    lv[i] = fun(*args)[0]
def _prc(target, kwargs):
    """Multiprocessing for 'save_x' functions."""
    start, mid, k_f, it = time.time(), 0., mp.Lock(), kwargs['it']
    kwargs['lock'] = k_f
    verbose = kwargs['verbose']
    kwargs['verbose'] = False; kwargs['ch_graph'] = False
    kwargs['it'] = 1
    l_prc = [None for i in range(it)]
    for i in range(it):
        l_prc[i] = mp.Process(target=target, kwargs=kwargs)
        l_prc[i].start(); mid = time.time()-start-mid
        prt(f"Starting: iteration {i+1}/{it}: {mid:.02f}s", verbose)
    for i, prc in enumerate(l_prc):
        mid = time.time()-start-mid
        prt(f"Waiting on {i}/{it}: {mid:0.2f}s", verbose)
        prc.join()
    prt(f"Finished: {time.time()-start:.02f}s", verbose)
def _load_gen(verbose=True, **kwargs):
    """Load an instance of 'Gen' and loads the pre-parsed data."""
    start = time.time()
    gen = Gen(f=""); gen.load_parsed("code/ofrom_gen.joblib")
    prt(f"Parsed: {time.time()-start:.02f}s", verbose)
    return gen
def _loop(strat, lim, loop, verbose, ch_graph, title):
    """Common code for passive/active."""
    c, y, xm = 0, [], np.floor(lim/1000)
    start = time.time()
    prt("Starting loops...", verbose)
    for acc_score in strat(lim):
        y.append(acc_score); c += 1
        prt(f"\tloop {c}/{loop}: {time.time()-start:.02f}s", verbose)
        start = time.time()
        if loop > 0 and c >= loop:
            break
    if ch_graph:
        x = [i*xm for i in range(len(y))]
        plt.title(title)
        plt.xlabel("Token count (thousands)")
        plt.ylabel("Accuracy score")
        plt.plot(x, y)
        plt.show()
    return y
def _sloop(strat, lim, it, loop, verbose, ch_graph, alpha, f, lock, title):
    """Common code for save_passive/active."""
    start = time.time()
    gen = _load_gen(verbose)
    for a in range(it):
        acc = strat(gen, lim, loop, verbose, False)
        prt(f"Save {a+1}/{it}: {time.time()-start:.02f}s., {gen.s} tokens", 
            verbose)
        if lock:    # parallel processing
            with lock:
                save_acc(f, acc)
        else:       # single use
            save_acc(f, acc)
    if ch_graph:
        acc = load_acc(f)
        _plt_ci(acc, np.floor(lim/1000), alpha, title)

    # Json #
    #------#
def load_json(f):
    """Loads a json."""
    if not os.path.isfile(f):
        return []
    with open(f, 'r', encoding="utf-8") as rf:
        return json.load(rf)
def save_acc(f, acc):
    """Stores the accuracy (list<float>) in a file."""
    if not os.path.isfile(f):   # new
        o_acc = [[a] for a in acc]
    else:                       # "append"
        o_acc = load_json(f)
        for a in range(len(acc)):
            o_acc[a].append(acc[a])
    with open(f, 'w', encoding="utf-8") as wf:
        json.dump(o_acc, wf)
def save_all(f, data):
    """Saves accuracy/tokens in a file.
       dict<'acc': list<float>, 
            'tok': dict<str: list<float>>
       >."""
    if not os.path.isfile(f):   # new
        o_data = {
            'acc': [[a] for a in data['acc']],
            'tok': data['tok']
        }
    else:
        o_data = load_json(f)
        for a in range(len(data['acc'])):
            o_data['acc'][a].append(data['acc'][a])
        for tn, tl in data['tok'].items():
            for a in range(len(tl)):
                o_data['tok'][tn][a].append(tl[a])
    with open(f, 'w', encoding="utf-8") as wf:
        json.dump(o_acc, wf)

    # Passive training #
    #------------------#
def passive(gen, lim=10000, loop=10, verbose=True, ch_graph=True):
    """Plots a single iterated passive training."""
    return _loop(gen.iter_passive, lim, loop, verbose, 
                 ch_graph, "Passive accuracy")
def save_passive(lim=10000, it=10, loop=10, verbose=True, ch_graph=False, 
                 alpha=0.95, f="passive_acc.json", lock=None, **kwargs):
    """Saves a repeated passive training at each iteration.
       'loop' < 0 exhausts all data."""
    _sloop(passive, lim, it, loop, verbose, ch_graph, alpha, f, lock,
           "Passive training")
def prc_passive(**kwargs):
    """Multiprocessing over 'save_passive'."""
    _prc(save_passive, kwargs)

    # Active training #
    #-----------------#
def active(gen, lim=10000, loop=10, verbose=True, ch_graph=True):
    """Plots a single (variable tokens) active training."""
    return _loop(gen.iter_active, lim, loop, verbose,
                 ch_graph, "Active accuracy")
def save_active(lim=10000, it=10, loop=10, alpha=0.95, verbose=True, 
                ch_graph=True, f="active_acc.json", lock=None, **kwargs):
    """Saves a repeated active (variable) training at each iteration.
       'loop' < 0 exhausts all data."""
    _sloop(active, lim, it, loop, verbose, ch_graph, alpha, f, lock,
           "Active training")
def prc_active(**kwargs):
    """Multiprocessing over 'save_active_v'."""
    _prc(save_active, kwargs)

    # Main #
    #------#
def regen(rf="code/ofrom_alt.joblib", wf="code/ofrom_gen.joblib"):
    """Rebuilds the data used by Gen."""
    gen = Gen(); gen.load_dataset("code/ofrom_alt.joblib")
    gen.save("code/ofrom_gen.joblib")
def plot_acc(f, lim=10000, alpha=0.95, title="Training", **kwargs):
    """Plot the accuracy after it has been generated/saved."""
    return _plt_ci(load_json(f), np.floor(lim/1000), alpha, title)
def plot_all(l_f=[], alpha=0.95, name="alt"):
    """Plots a graph with both learning curves."""
    l_f = [
        "json/pas_10k_10.json", f"json/orc_10k_10.json",
        "json/pas_10k_10.json", f"json/act_10k_10.json"
    ] if not l_f else l_f # json files to plot, by pairs
    l_tmp = [
        "passive", "oracle",
        "passive", "active"
    ] # custom labels
    lf = len(l_f)//2
    fig, ax = plt.subplots(1, lf, figsize=(10, 5))
    for a in range(0, lf): # for each subplot
        lim = 10000          # temporary
        # lim = 10**(a+4)    # still manual, up to 1mn
        y1, y2 = load_json(l_f[a*2]), load_json(l_f[(a*2)+1]) # y-axes
        ln, lm = max(len(y1), len(y2)), min(len(y1[0]), len(y2[0]))
        mul = np.floor(lim/1000) if lim >= 1000 else np.floor(lim)
        x = [(i+1)*mul for i in range(ln)]                    # x-axis
        _plt(x, y1, alpha, 'b', l_tmp[a*2], ax[a])
        _plt(x, y2, alpha, 'r', l_tmp[(a*2)+1], ax[a])
        txt = f"Comparison ({int(lim/1000)}k)" if lim >= 1000 else \
              f"Comparison ({lim})"
        ax[a].set_title(txt)
        txt = "Token count"
        txt = txt+" (thousands)" if lim >= 1000 else txt
        ax[a].set_xlabel(txt)
        ax[a].set_ylabel(f"Accuracy score ({lm})")
        ax[a].legend()
    fig.tight_layout()
    plt.savefig("img/summary.png")
    
def _typ(v):
    """Converts type the old way."""
    d_typ = {'None': None, 'True': True, 'true':True,
           'False':False, 'false': False}
    if v in d_typ:                      # boolean or NoneType
        return d_typ[v]
    elif re.match(r"^[+-]?\d*$", v):    # int
        return int(v)
    try:                                # float (regex too complex)
        v = float(v); ch_typ = True
    except ValueError:
        pass
    try:                                # list/dict
        v = json.loads(v)
    except json.JSONDecodeError:        # string
        return v
def _args(args):
    """Updates default parameters using sys.argv."""
    d_func = {
        'plot': plot_acc,
        'passive': save_passive,
        'active': save_active
    }
    l_args = ['func', 'lim', 'it', 'loop', 'alpha', 
              'verbose', 'ch_graph', 
              'f', 'title'] # key list
    d_args = {
        'func': None,
        'lim': 1000, 'it': 10, 'loop': 100, 'alpha': 0.95, 
        'verbose': True, 'ch_graph': False,
        'f': "passive_acc.json", "title": "Training"
    }                   # default values
    for a, arg in enumerate(args):
        if "=" in arg:  # kwarg
            k, v = arg.split("=",1)
            d_args[k] = _typ(v.replace("\"", "").replace("'", ""))
        else:           # arg
            d_args[l_args[a]] = _typ(arg)
    if isinstance(d_args['func'], str):
        d_args['func'] = d_func.get(d_args['func'], None)
    return d_args
if __name__ == "__main__":
    kwargs = _args(sys.argv[1:])    # get kwargs
    if ('func' in kwargs) and kwargs['func'] != None:
        kwargs['func'](**kwargs)    # explicit function call
        sys.exit()
    plot_all([], 0.95, "act")
    # regen("code/ofrom_alt.joblib", "code/ofrom_gen.joblib")
    # gen = _load_gen()
    # gen.optimize()
    sys.exit()