<<<<<<< HEAD
from ofrom_gen import prep_sequ, Ofrom_gen
import matplotlib.pyplot as plt
from scipy.stats import t as sci_t
import numpy as np
import threading as thr
import joblib, time

    # Support #
    #---------#
def prt(txt):
    """Overwrites current line."""
    print("\r"+" "*100+"\r"+txt, end="")
def _plt_ci(y, xm, alpha, title):
    """Plots for non-fixed CI functions."""
    x = [i*xm for i in range(len(y[0]))] # /!\ it > 0
    my, cuy, cdy = [], [], []
    for el in y:                    # manual build of mean +/- var
        my.append(np.mean(el))
        mod = np.std(el)/np.sqrt(len(el))
        v_el = sci_t.interval(alpha, df=len(el)-1, loc=my[-1],
                              scale=mod)
        cuy.append(v_el[1])
        cdy.append(v_el[0])
    plt.plot(x, my, color='b', linewidth=2)
    plt.plot(x, cuy, linestyle='--', color='b', linewidth=1)
    plt.plot(x, cdy, linestyle='--', color='b', linewidth=1)
    plt.title(title)
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    title = title.replace(" ", "_").lower()+".png"
    plt.savefig(title)

    # Passive training #
    #------------------#
def passive(lim=10000, loop=10, verbose=True):
    """Plots a single iterated passive training."""
    c, y, xm = 0, [], np.floor(lim/1000)
    start = time.time()
    for acc_score, l_toks in gen.iter_passive(lim=lim, nb=10):
        y.append(acc_score); c += 1
        if verbose:
            print(f"No-iteration, loop {c}: {time.time()-start:.02f}s")
            start = time.time()
        if loop > 0 and c >= loop:
            break
    x = [i*xm for i in range(len(y))]
    plt.title("Passive accuracy")
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    plt.plot(x, y)
    plt.show()
def passive_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_batch=5,
               verbose=True):
    """Plots a repeated passive training.
       'loop' must be >= 0."""
    nb_thr = it*loop*nb_batch; gen.pool_set(nb_thr)
    acc, xm = [[] for b in range(loop)], np.floor(lim/1000)
    for a in range(it):
        gen.reset(); l_sequs, X, y = [], [], []
        for b in range(loop):
            acc[b].append([])
            ln_sequs = gen.sel_rand(lim)
            for s in ln_sequs:                          # X/y training
                nx, ny = prep_sequ(s); X.append(nx); y.append(ny)
            l_sequs = l_sequs+ln_sequs                  # loop batch
            l_sub = gen._cross_tr(l_sequs, nb_batch)    # sub-batches
            for c in range(len(l_sub)):                 # add to pool
                acc[b][-1].append(gen.pool_add(gen._tr,
                                  args=(X, y, l_sub[c], {})))
            prt(f"Starting: iteration {a+1}/{it}, loop {b+1}/{loop}, "+
                f"{len(gen.pool_res)}/{nb_thr} workers.")
    gen.pool_close(); start = mid = time.time()
    for b, iacc in enumerate(acc):                      # get scores
        for i, lacc in enumerate(iacc):
            for j, res in enumerate(lacc):
                end = time.time()-mid if time.time()-mid < 0.1 else end
                prt(f"Waiting on {b*i*j}/{nb_thr}, "+
                    f"last: {end:.02f}s")
                acc[b][i][j] = res.get()[0]
                mid = time.time()
            acc[b][i] = np.mean(acc[b][i])
    gen.pool_rem()
    prt(f"Plotting: {time.time()-start:.02f}s total")
    for i, lacc in enumerate(acc):                      # avg accuracy
        for j, bacc in enumerate(lacc):
            acc[i][j] = bacc/nb_batch
    _plt_ci(acc, xm, alpha, "Passive training")
        
    # Active training #
    #-----------------#
def get_active(ch_fixed=False, lim=10000, loop=10, nb_toks=10,
               g_toks=None):
    """Common part between fixed/variable active training."""
    c, x, y, l_y = 0, [], [], [[] for a in range(nb_toks)]
    for acc_score, l_toks in gen.iter_active(lim=lim, nb=nb_toks, 
                                 ch_fixed=ch_fixed, g_toks=g_toks):
        c += 1; x.append(c*10); y.append(acc_score)
        for i in range(len(l_y)): # confidence scores
            l_y[i].append(l_toks[i][1])
        print(f"\tLoop: {c}")
        for tok in l_toks:
            print("\t", tok)
        if loop > 0 and c >= loop:
            break
    return x, y, l_y, [tok[0] for tok in l_toks]
def active_fixed(lim=10000, loop=10, nb_toks=10, g_toks=None):
    """Plots a single (fixed tokens) active training."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("Active training (fixed)")
    ax[0].set_xlabel("Token count (thousands)")
    ax[0].set_ylabel("Accuracy score")
    ax[1].set_title("Token confidence score")
    ax[1].set_xlabel("Token count (thousands)")
    ax[1].set_ylabel("Confidence score")
    x, y, l_y, l_lgd = get_active(True, lim, loop, nb_toks, g_toks)
    ax[0].plot(x, y)
    for i, vy in enumerate(l_y):
        ax[1].plot(x, vy, label=l_lgd[i])
    ax[1].legend()
    plt.show()
def active_variable(lim=10000, loop=10, nb_toks=10):
    """Plots a single (variable tokens) active training."""
    plt.title("Active training (variable)")
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    x, y, l_y, l_lgd = get_active(False, lim, loop, nb_toks)
    plt.plot(x, y)
    plt.show()
def active_var_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_toks=10):
    """Plots a repeated (variable tokens) active training."""
    y, xm = [], np.floor(lim/1000)
    for i in range(it):
        print(f"Iteration: {i+1}")
        ix, iy, l_y, l_lgd = get_active(False, lim, nb_toks)
        y.append(iy)
    _plt_ci(y, xm, alpha, "Active training (variable)")

if __name__ == "__main__":
    gen = Ofrom_gen(f="")
    gen.load_parsed("ofrom_gen.joblib")    # load pre-parsed data
    prt("parsed")
    
    # crf, sc, d = gen.train_passive(False)
    # joblib.dump(crf, "ofrom_crf.joblib", compress=5)
    # passive(lim=10000, loop=10)
    passive_ci(lim=10000, it=20, loop=10, alpha=0.95)
    g_toks = [
        ('leur', 0.1),
        ('suivant', 0.5),
        ('comment', 0.3),
        ('sinon', 0.1),
        ('contre', 0.1),
        ('depuis', 0.1),
        ('vu', 0.2),
        ('sauf', 0.3),
        ('la', 0.2),
        ('passé', 0.7)
    ]
    # active_fixed(lim=10000, loop=10, nb_toks=10, g_toks=g_toks)
    # active_variable(lim=10000, loop=10, nb_toks=10)
    # active_var_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_toks=10)
=======
from ofrom_gen import prep_sequ, Ofrom_gen
import matplotlib.pyplot as plt
from scipy.stats import t as sci_t
import numpy as np
import threading as thr
import joblib, time

    # Support #
    #---------#
def prt(txt):
    """Overwrites current line."""
    print("\r"+" "*100+"\r"+txt, end="")
def _plt_ci(y, xm, alpha, title):
    """Plots for non-fixed CI functions."""
    x = [i*xm for i in range(len(y[0]))] # /!\ it > 0
    my, cuy, cdy = [], [], []
    for el in y:                    # manual build of mean +/- var
        my.append(np.mean(el))
        mod = np.std(el)/np.sqrt(len(el))
        v_el = sci_t.interval(alpha, df=len(el)-1, loc=my[-1],
                              scale=mod)
        cuy.append(v_el[1])
        cdy.append(v_el[0])
    plt.plot(x, my, color='b', linewidth=2)
    plt.plot(x, cuy, linestyle='--', color='b', linewidth=1)
    plt.plot(x, cdy, linestyle='--', color='b', linewidth=1)
    plt.title(title)
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    title = title.replace(" ", "_").lower()+".png"
    plt.savefig(title)

    # Passive training #
    #------------------#
def passive(lim=10000, loop=10, verbose=True):
    """Plots a single iterated passive training."""
    c, y, xm = 0, [], np.floor(lim/1000)
    start = time.time()
    for acc_score, l_toks in gen.iter_passive(lim=lim, nb=10):
        y.append(acc_score); c += 1
        if verbose:
            print(f"No-iteration, loop {c}: {time.time()-start:.02f}s")
            start = time.time()
        if loop > 0 and c >= loop:
            break
    x = [i*xm for i in range(len(y))]
    plt.title("Passive accuracy")
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    plt.plot(x, y)
    plt.show()
def passive_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_batch=5,
               verbose=True):
    """Plots a repeated passive training.
       'loop' must be >= 0."""
    nb_thr = it*loop*nb_batch; gen.pool_set(nb_thr)
    acc, xm = [[] for b in range(loop)], np.floor(lim/1000)
    for a in range(it):
        gen.reset(); l_sequs, X, y = [], [], []
        for b in range(loop):
            acc[b].append([])
            ln_sequs = gen.sel_rand(lim)
            for s in ln_sequs:                          # X/y training
                nx, ny = prep_sequ(s); X.append(nx); y.append(ny)
            l_sequs = l_sequs+ln_sequs                  # loop batch
            l_sub = gen._cross_tr(l_sequs, nb_batch)    # sub-batches
            for c in range(len(l_sub)):                 # add to pool
                acc[b][-1].append(gen.pool_add(gen._tr,
                                  args=(X, y, l_sub[c], {})))
            prt(f"Starting: iteration {a+1}/{it}, loop {b+1}/{loop}, "+
                f"{len(gen.pool_res)}/{nb_thr} workers.")
    gen.pool_close(); start = mid = time.time()
    for b, iacc in enumerate(acc):                      # get scores
        for i, lacc in enumerate(iacc):
            for j, res in enumerate(lacc):
                end = time.time()-mid if time.time()-mid < 0.1 else end
                prt(f"Waiting on {b*i*j}/{nb_thr}, "+
                    f"last: {end:.02f}s")
                acc[b][i][j] = res.get()[0]
                mid = time.time()
            acc[b][i] = np.mean(acc[b][i])
    gen.pool_rem()
    prt(f"Plotting: {time.time()-start:.02f}s total")
    for i, lacc in enumerate(acc):                      # avg accuracy
        for j, bacc in enumerate(lacc):
            acc[i][j] = bacc/nb_batch
    _plt_ci(acc, xm, alpha, "Passive training")
        
    # Active training #
    #-----------------#
def get_active(ch_fixed=False, lim=10000, loop=10, nb_toks=10,
               g_toks=None):
    """Common part between fixed/variable active training."""
    c, x, y, l_y = 0, [], [], [[] for a in range(nb_toks)]
    for acc_score, l_toks in gen.iter_active(lim=lim, nb=nb_toks, 
                                 ch_fixed=ch_fixed, g_toks=g_toks):
        c += 1; x.append(c*10); y.append(acc_score)
        for i in range(len(l_y)): # confidence scores
            l_y[i].append(l_toks[i][1])
        print(f"\tLoop: {c}")
        for tok in l_toks:
            print("\t", tok)
        if loop > 0 and c >= loop:
            break
    return x, y, l_y, [tok[0] for tok in l_toks]
def active_fixed(lim=10000, loop=10, nb_toks=10, g_toks=None):
    """Plots a single (fixed tokens) active training."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("Active training (fixed)")
    ax[0].set_xlabel("Token count (thousands)")
    ax[0].set_ylabel("Accuracy score")
    ax[1].set_title("Token confidence score")
    ax[1].set_xlabel("Token count (thousands)")
    ax[1].set_ylabel("Confidence score")
    x, y, l_y, l_lgd = get_active(True, lim, loop, nb_toks, g_toks)
    ax[0].plot(x, y)
    for i, vy in enumerate(l_y):
        ax[1].plot(x, vy, label=l_lgd[i])
    ax[1].legend()
    plt.show()
def active_variable(lim=10000, loop=10, nb_toks=10):
    """Plots a single (variable tokens) active training."""
    plt.title("Active training (variable)")
    plt.xlabel("Token count (thousands)")
    plt.ylabel("Accuracy score")
    x, y, l_y, l_lgd = get_active(False, lim, loop, nb_toks)
    plt.plot(x, y)
    plt.show()
def active_var_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_toks=10):
    """Plots a repeated (variable tokens) active training."""
    y, xm = [], np.floor(lim/1000)
    for i in range(it):
        print(f"Iteration: {i+1}")
        ix, iy, l_y, l_lgd = get_active(False, lim, nb_toks)
        y.append(iy)
    _plt_ci(y, xm, alpha, "Active training (variable)")

if __name__ == "__main__":
    gen = Ofrom_gen(f="")
    gen.load_parsed("ofrom_gen.joblib")    # load pre-parsed data
    prt("parsed")
    
    # crf, sc, d = gen.train_passive(False)
    # joblib.dump(crf, "ofrom_crf.joblib", compress=5)
    # passive(lim=10000, loop=10)
    passive_ci(lim=10000, it=20, loop=10, alpha=0.95)
    g_toks = [
        ('leur', 0.1),
        ('suivant', 0.5),
        ('comment', 0.3),
        ('sinon', 0.1),
        ('contre', 0.1),
        ('depuis', 0.1),
        ('vu', 0.2),
        ('sauf', 0.3),
        ('la', 0.2),
        ('passé', 0.7)
    ]
    # active_fixed(lim=10000, loop=10, nb_toks=10, g_toks=g_toks)
    # active_variable(lim=10000, loop=10, nb_toks=10)
    # active_var_ci(lim=10000, it=10, loop=10, alpha=0.95, nb_toks=10)
>>>>>>> d57c33111a14b2c53ee01709ba0c303aae31a6b5
    wait = input()