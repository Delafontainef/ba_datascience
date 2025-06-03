""" 03.06.2025
Simulates active learning.

Supports command calls:
    python test.py function [args/kwargs]
With supported functions:
- 'exp': runs the experiment ('it' times). 
The list of parameters (args/kwargs) is in the '_args' function,
along with their default values.
            
An experiment is:
- selecting a reference dataset (fixed or random: 'fixed' parameter).
- selecting an initial subset (fixed or random: 'fixed' parameter).
- repeating a loop 'loop' times:
|- train the model on the subset
|- evaluate the model on the reference dataset
|- select more data for the subset
|-- at random (passive learning)
|-- with a query strategy (active learning) 
The set of accuracy scores is saved in a file ('f' parameter).
See 'ofrom_gen.py' for the strategies.

The 'plot' function loads the saved accuracies and plots with CIs.

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

    # Json #
    #------#
def load_json(f):
    """Loads a json."""
    if not os.path.isfile(f):
        return []
    with open(f, 'r', encoding="utf-8") as rf:
        return json.load(rf)
def save_json(f, acc):
    """Stores the accuracy (list<float>) in a file."""
    if not os.path.isfile(f):   # new
        o_acc = [[a] for a in acc]
    else:                       # "append"
        o_acc = load_json(f)
        for a in range(len(acc)):
            o_acc[a].append(acc[a])
    with open(f, 'w', encoding="utf-8") as wf:
        json.dump(o_acc, wf)

    # Training #
    #----------#
def load_gen(**kwargs):
    """Load an instance of 'Gen' with pre-parsed data."""
    start = time.time()
    path = kwargs.get('gen_path', 'code/ofrom_gen.joblib')
    verbose = kwargs.get('verbose', True)
    gen = Gen(f=""); gen.load_parsed(path)
    prt(f"Parsed: {time.time()-start:.02f}s", verbose)
    return gen
def loop(gen, c_it=-1, **kwargs):
    """Minimal (select-train-evaluate) loop function."""
    loop, l_acc, c, start = kwargs.get('loop', 10), [], 0, time.time()
    verbose = kwargs.get('verbose', True)
    for acc_score in gen.iter(**kwargs):            # loop
        l_acc.append(acc_score); c += 1             # append accuracy score
        txt = (f"loop {c}/{loop}: {time.time()-start:.02f}s"+
               f", {gen.s} tokens")
        txt = "\t"+txt if c_it < 0 else f"\tit {c_it}/{kwargs['it']} "+txt
        prt(txt, verbose)                           # print
        start = time.time()
        if loop > 0 and c >= loop:                  # check if done
            break
    return l_acc
def experiment(**kwargs):
    """Code to run one replication."""
    gen, it = load_gen(**kwargs), kwargs.get('it', 10)
    f = kwargs.get('f', "")                         # json file path
    for c in range(it):
        gen.reset(**kwargs)
        l_acc = loop(gen, c_it=c+1, **kwargs)
        save_json(f, l_acc)
def oracle(**kwargs):
    """For a fixed reference dataset, finds the most accurate subset."""
        # unpack
    lim, features = kwargs.get('lim', 100000), kwargs.get('features', False)
    f, loop = kwargs.get('f', "oracle_ind.json"), kwargs.get('loop', 10)
    verbose, fixed = kwargs.get('verbose', True), kwargs.get('fixed', True)
        # generate
    gen = load_gen(**kwargs)
    jd = {'acc':[], 'ind':[], 'len':[]}             # json in-flight data
    old_s, l_acc, llim = gen.s, [], lim//loop
    for acc, l_ind in gen.oracle(lim, features, fixed, verbose):
        if gen.s//llim > old_s//llim:               # add to accuracy
            old_s = gen.s; l_acc.append(acc)
        jd['acc'].append(acc); jd['len'].append(gen.s); jd['ind'] = l_ind
        with open("oracle.json", 'w', encoding="utf-8") as wf: # overwrite
            json.dump(jd, wf)
    while len(l_acc) < loop:                        # ensure last loop.s
        l_acc.append(acc)
    save_json(f, l_acc)
    
def plot(l_master, wf="img/summary.png", alpha=0.95, fixed_y=False):
    """Code to plot the jsons."""
    nr = len(l_master); nc = min(3, nr); nr = int(np.ceil(nr/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 4*nr), squeeze=False)
    axes = axes.flatten()
    colors = ['b', 'r', 'k', 'g', 'c', 'm', 'y']
    yrows = {r: [float('inf'), float('-inf')] for r in range(nr)}
    for i, ax in enumerate(axes):
        if i >= len(l_master):                  # unused
            ax.axis('off'); continue
        l_f, l_lgd, lim = l_master[i]           # unpack
        ch_k = True if lim >= 1000 else False
        mul = np.floor(lim/1000) if ch_k else np.floor(lim)
        ln, lm, l_y = -1, np.inf, []
        for j, f in enumerate(l_f):
            y = load_json(f); l_y.append(y)
            ln = len(y) if len(y) > ln else ln
            lm = min(lm, len(y[0]))
        x = [(i+1)*mul for i in range(ln)]      # x-axis
        r =  i//nc                              # row index
        for j, y in enumerate(l_y):             # plot each line
            _plt(x, y, alpha, colors[j], l_lgd[j], ax)
            if fixed_y:                         # track y-axis scale
                y_min, y_max = yrows[r]
                yrows[r] = [min(y_min, np.min(y)), max(y_max, np.max(y))]
        txt = f"Comparison ({int(lim/1000)}k)" if ch_k else \
              f"Comparison ({lim})"
        ax.set_title(txt)
        txt = "Token count"
        txt = txt+" (thousands)" if ch_k else txt
        ax.set_xlabel(txt)
        ax.set_ylabel(f"Accuracy score ({lm})")
        ax.legend()
    if fixed_y:                                 # set y-axis scale
        for i, axi in enumerate(axes[:len(l_master)]):
            r = i//nc
            y_min, y_max = yrows[r]
            axi.set_ylim(y_min, y_max)
    fig.tight_layout()
    plt.savefig(wf)

    # Others #
    #--------#
def regen(rf="code/ofrom_alt.joblib", wf="code/ofrom_gen.joblib"):
    """Rebuilds the data used by Gen."""
    gen = Gen(); gen.load_dataset(rf); gen.save(wf)
def decompose(rf="code/ofrom_gen.joblib", wf="code/fake.joblib",
              nb_toks=5, lim=200000):
    """Makes 1 file per sequence, for a 'lim' dataset (-1 for all).
       'nb_toks' is the max tokens per sequence (-1 for all)."""
    gen = Gen(); gen.load_dataset(rf); gen.decompose(wf, nb_toks, lim)
def optimize(**kwargs):
    """Defines c1/c2 hyperparameters."""
    rf = kwargs.get('gen_path', 'code/ofrom_gen.joblib')
    lim, features = kwargs.get('lim', 100000), kwargs.get('features', False)
    verbose = kwargs.get('verbose', True)
    gen = Gen(); gen.load_parsed(rf)
    if features:
        gen.add_ft()
    return gen.optimize(lim, verbose)

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
def _args(args=[]):
    """Updates default parameters using sys.argv."""
    d_func = {
        'exp': experiment,
        'oracle': oracle
    }
    d_args = {
        'func': None, 'strat': "file_conf",
        'lim': 1000, 'it': 10, 'loop': 100, 'alpha': 0.95, 
        'verbose': True, 'ch_graph': False,
        'f': "passive_acc.json", 'title': "Training",
        'avg': "avg", 'features': False, 'data_size': -1,
        'fixed': False, 'ref_size': 100000
    }                   # default values
    k_args = list(d_args.keys())
    for a, arg in enumerate(args):
        if "=" in arg:  # kwarg
            k, v = arg.split("=",1)
            d_args[k] = _typ(v.replace("\"", "").replace("'", ""))
        else:           # arg
            d_args[k_args[a]] = _typ(arg)
    if isinstance(d_args['func'], str):
        d_args['func'] = d_func.get(d_args['func'], None)
    return d_args
if __name__ == "__main__":
    kwargs = _args(sys.argv[1:])    # get kwargs
    if ('func' in kwargs) and kwargs['func'] != None:
        kwargs['func'](**kwargs)    # explicit function call
        sys.exit()
        # default function
    l_master = [
        [
            ["paslim_10k_10.json", "actlim_10k_10.json"], 
            ["passive", "active"], 
            10000
        ]
    ]
    plot(l_master)
    sys.exit()