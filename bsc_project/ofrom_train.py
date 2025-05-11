""" 10.05.2025
Simulates passive/active training.

Supports command calls:
    python test.py function [parameters]
With supported functions:
- 'plot': Plots the accuracy based on a pre-saved json file.
          ('plot_acc' function)
- 'passive': Tests conventional (passive) training.
             ('save_passive' function)
- 'active': Tests active training.
            The strategy is hardcoded in the 'Gen' class.
            ('save_active_v' function)
The list of parameters (args/kwargs) is in the '_args' function,
along with their default values.
            
When testing, we:
- increment 'loop' times an initially empty subset 
  by batches of 'lim' tokens
- train it 5 times in a cross-validation way and average the accuracy score
- passive training selects the next batch at random
- active training selects based on tokens' confidence level
- we repeat that process 'it' times and save the accuracy each time 
  in file 'f'.

When plotting, we load the saved accuracy and plot its average plus CI.

Note: A version of active training exists with fixed tokens.
      It was abandoned for obvious reasons but allowed plotting how a set
      of tokens' confidence score evolved.
Note: All training functions are in 'ofrom_crf'
      and handled by the 'Gen' class.
Note: Attempts at multithreading have proved slower than a continuous
      version. The cross-training is still handled by threads in 
      the Gen class.
Note: Tokens are grouped in 'sequ' (sequences) for CRF training.
      Sequences are grouped in 'file', the minimal shared unit.
"""

import sys, os, re, time, json, joblib
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                'code')) # ...
from ofrom_gen import prep_sequ, Gen
import matplotlib.pyplot as plt
from scipy.stats import t as sci_t
import numpy as np
import threading as thr
import multiprocessing as mp

    # Support #
    #---------#
def prt(txt, verbose=True):
    """Overwrites current line."""
    if verbose:
        print("\r"+" "*100+"\r"+txt, end="")
def _plt_ci(y, xm, alpha, title, ch_graph=True):
    """Plots for non-fixed CI functions."""
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
def _load_gen(verbose=True, **kwargs):
    """Load an instance of 'Gen' and loads the pre-parsed data."""
    start = time.time()
    gen = Gen(f=""); gen.load_parsed("ofrom_gen.joblib")
    prt(f"Parsed: {time.time()-start:.02f}s", verbose)
    return gen
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
def passive(lim=10000, loop=10, nb_batch=5, verbose=True, ch_graph=True):
    """Plots a single iterated passive training."""
    c, y, xm = 0, [], np.floor(lim/1000)
    start = time.time()
    try:
        gen.reset()
    except Exception:
        gen = _load_gen(verbose)
    prt("Starting loops...", verbose)
    for acc_score in gen.iter_passive(lim=lim, nb_batch=nb_batch):
        y.append(acc_score); c += 1
        prt(f"\tloop {c}/{loop}: {time.time()-start:.02f}s", verbose)
        start = time.time()
        if loop > 0 and c >= loop:
            break
    if ch_graph:
        x = [i*xm for i in range(len(y))]
        plt.title("Passive accuracy")
        plt.xlabel("Token count (thousands)")
        plt.ylabel("Accuracy score")
        plt.plot(x, y)
        plt.show()
    return y
def save_passive(lim=10000, it=10, loop=10, alpha=0.95, nb_batch=5,
                 verbose=True, ch_graph=False, f="passive_acc.json",
                 lock=None, **kwargs):
    """Saves a repeated passive training at each iteration.
       'loop' < 0 exhausts all data."""
    start = time.time()
    for a in range(it):
        acc = passive(lim, loop, nb_batch, verbose, False)
        prt(f"Save {a+1}/{it}: {time.time()-start:.02f}s.", verbose)
        if lock:    # parallel processing
            with lock:
                save_acc(f, acc)
        else:       # single use
            save_acc(f, acc)
    if ch_graph:
        acc = load_acc(f)
        _plt_ci(acc, np.floor(lim/1000), alpha, "Passive training")
def prc_passive(**kwargs):
    """Multiprocessing over 'save_passive'."""
    _prc(save_passive, kwargs)

    # Active training #
    #-----------------#
def get_active(ch_fixed=False, lim=10000, loop=10, nb_toks=10, g_toks=None):
    """Common part between fixed/variable active training."""
    c, x, y, l_y = 0, [], [], [[] for a in range(nb_toks)]
    try:
        gen.reset()
    except Exception:
        gen = _load_gen(verbose)
    for acc_score in gen.iter_active(lim=lim, nb=nb_toks, 
                                     ch_fixed=ch_fixed, g_toks=g_toks):
        c += 1; x.append(c*10); y.append(acc_score)
        for i in range(len(l_y)): # confidence scores
            l_y[i].append(gen.toks[i][1])
        print(f"\tLoop: {c}")
        for tok in gen.toks:
            print("\t", tok)
        if loop > 0 and c >= loop:
            break
    return x, y, l_y, [tok[0] for tok in l_toks]
def active_fixed(lim=10000, loop=10, nb_toks=10, g_toks=None):
    """Plots a single (fixed tokens) active training."""
    x, y, l_y, l_lgd = get_active(True, lim, loop, nb_toks, g_toks)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("Active training (fixed)")
    ax[0].set_xlabel("Token count (thousands)")
    ax[0].set_ylabel("Accuracy score")
    ax[1].set_title("Token confidence score")
    ax[1].set_xlabel("Token count (thousands)")
    ax[1].set_ylabel("Confidence score")
    ax[0].plot(x, y)
    for i, vy in enumerate(l_y):
        ax[1].plot(x, vy, label=l_lgd[i])
    ax[1].legend()
    plt.show()
    return x, y, l_y, l_lgd
def active_variable(lim=10000, loop=10, nb_toks=10):
    """Plots a single (variable tokens) active training."""
    x, y, l_y, l_lgd = get_active(False, lim, loop, nb_toks)
    plt.title("Active training (variable)")
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    plt.plot(x, y)
    plt.show()
def save_active_v(lim=10000, it=10, loop=10, alpha=0.95, nb_batch=5,
                  nb_toks=10, verbose=True, ch_graph=True,
                  f="active_acc.json", lock=None, **kwargs):
    """Saves a repeated active (variable) training at each iteration.
       'loop' < 0 exhausts all data."""
    start = time.time()
    try:
        gen.reset()
    except Exception:
        gen = _load_gen(verbose)
    for a in range(it):
        x, acc, l_acc, l_toks = get_active(False, lim, loop, nb_toks, None)
        prt(f"Save {a}/{it}: {time.time()-start:.02f}s.", verbose)
        if lock:    # parallel processing
            with lock:
                save_acc(f, acc)
        else:       # single use
            save_acc(f, acc)
    if ch_graph:
        acc = load_acc(f)
        _plt_ci(acc, np.floor(lim/1000), alpha, "Active training (variable)")
def prc_active_v(**kwargs):
    """Multiprocessing over 'save_active_v'."""
    _prc(save_active_v, kwargs)

    # Main #
    #------#
def plot_acc(f, lim=10000, alpha=0.95, title="Training", **kwargs):
    """Plot the accuracy after it has been generated/saved."""
    return _plt_ci(load_json(f), np.floor(lim/1000), alpha, title)
    
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
        'active': save_active_v
    }
    l_args = ['func', 'lim', 'it', 'loop', 'alpha', 'nb_batch',
              'verbose', 'ch_graph', 'nb_toks', 'g_toks',
              'f', 'title'] # key list
    d_args = {
        'func': None,
        'lim': 1000, 'it': 10, 'loop': 100, 'alpha': 0.95, 'nb_batch': 5,
        'verbose': True, 'ch_graph': False,
        'nb_toks': 10, 'g_toks': None,
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
    if (not 'func' in kwargs) or kwargs['func'] == None:
        sys.exit()
    kwargs['func'](**kwargs)        # passive training by default