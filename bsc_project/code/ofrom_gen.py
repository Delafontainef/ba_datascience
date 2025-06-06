""" 03.06.2025
A "generator" class to handle the dataset, notably to iterate over its content.
The main method is:
- iter:           selects reference dataset and initial subset,
                  then iterates over each selection 'loop' times.
In detail:
- load_dataset:   loads raw data and parses it
- load_parsed:    loads pre-parsed data
- reset:          called before iterating
|- set_size:      limits the size of the available data (-1 to ignore)
|- set_ref:       sets aside a reference dataset (for evaluation)
- it:             selects for the subset, trains and evaluates
|- select:        selects 'lim' tokens by files, using strategy 'strat'
|- train:         trains and evaluates on the subset
|-- tr:           trains a model
|-- pred:         evaluates (by default on reference dataset)

The strategies (for file selection) are:
- "rand":         at random
- "last":         always the last file
- "100":          from index 100 onward
- "file_conf":    average confidence score over the entire file's content
- "file_dvrs":    adds proportion of tokens to file_conf
- "file_orcl":    file's accuracy score ("oracle" strategy)
More modifiers:
- "features":     whether to add features to the CRF model ('add_ft()')
- "avg":          what average to use for file_conf/dvrs
|- "avg"          standard average
|- "macro"        macro average (average by class)
|- "entropy"      entropy
- "fixed":        whether the reference dataset and initial subset are
                  random (False) or not (True).
It is also possible to limit the amount of data ('data_size') and control the 
size of the reference dataset ('ref_size').

More methods:
- decompose:      turns the parsed data into a sequence per file
- optimize:       finds optimal c1/c2 hyperparameters
- oracle:         trains the model on each file, selects the highest 
                  highest accuracy score, for each file added to the subset.
                  Yields the accuracy score and list of file indices. 

Note: CRF methods are imported from 'ofrom_crf.py'.
      This script should contain no scikit-learn operation directly.
"""

from ofrom_crf import train, predict_one, acc_score, Featurer
from scipy.stats import entropy
from collections import Counter
import pandas as pd
import numpy as np
import os, re, time, json, heapq, joblib

    # From 'ofrom_pos.py' #
    #---------------------#
def read(f):
    """Reads the database.
       Edited to remove zipfile."""
    if not os.path.isfile(f):                   # no file
        return pd.DataFrame()
    f = os.path.abspath(f)
    d, file = os.path.split(f); fi, ext = os.path.splitext(file)
    d_e = {
        '.joblib': joblib.load, '.xlsx': pd.read_excel, '.csv': pd.read_csv,
        '.json': json.load
    }
    if ext.lower() in d_e:                      # based on extension
        return d_e[ext.lower()](f)
    return pd.DataFrame()  
def prep_sequ(sequ):
    """Turns sequence into X/y parts."""
    return [{'token': s['token']} for s in sequ], [s['pos'] for s in sequ]
def iter_sequ(f="ofrom_alt.joblib", s=0, lim=-1, ch_prep=False):
    """Iterates over the dataset to retrieve sequences of tokens (IPUs).
       Removes reserved symbols, truncations and short pauses."""
    l_tmp, df = [], read(f); dc = list(df.columns)
    onfile, onspk = "", ""
    r_sym = r"([#@%]|[/-]( |$))"
    lr = len(df.index) if lim <= 0 else s+lim
    for i in range(s, lr):              # for each token...
        tok, nfile = df.iloc[i]['token'], df.iloc[i]['file']
        nspk = df.iloc[i]['speaker']
        if (l_tmp and (onfile != nfile or onspk != nspk)): # new file/speaker
            onfile = nfile; onspk = nspk
            yield prep_sequ(l_tmp) if ch_prep else l_tmp; l_tmp = []
        if "_" in tok:                  # pauses
            deb, end = df.iloc[i]['start'], df.iloc[i]['end']
            if end-deb < 0.5:           # ignore short pauses
                continue
            elif l_tmp:                 # yield sequence
                yield prep_sequ(l_tmp) if ch_prep else l_tmp
                l_tmp = []
            continue
        elif re.search(r_sym, tok):     # reserved symbols / truncation
            continue
        l_tmp.append({k: df.iloc[i][k] for k in dc})
    if l_tmp:                           # last loop
        yield prep_sequ(l_tmp) if ch_prep else l_tmp

    # Generator #
    #-----------#
class Gen:
    """Handles 'file' partition of the dataset and training iteration.
    
    At its core, it's a list of files and a list containing their indexes.
    For iteration, that list is popped each time a file is selected.
    """
    
    def __init__(self, f="", s=0, lim=-1, keep=[], mt=1., mf=0., mc=0.):
            # operations
        self.keep = ['ADV', 'CON', 'DET', 'PRO', 'PRP'] if not keep else keep
            # data
        self.files, self.ref = [], []
        self.load_dataset(f, s, lim)        # fills the above
        self._ind = [i for i in range(len(self.files))] # file selection
        self.s = 0                          # subset size in tokens
        self.crf, self.acc = None, -1.      # CRF model and accuracy
        self.prf = []                       # prediction for active learning
        self.tok = {}                       # for diversity
        self.c1, self.c2 = 0.0127, 0.023    # hyperparameters
        self.strat = {
            "rand": (self._pick_rand, None),
            "last": (self._pick_last, None),
            "100": (self._pick_100, None),
            "toks": (self._pick_toks, self._pred_tokens),
            "conf": (self._pick_file, self._pred_files),
            "dvrs": (self._pick_file, self._pred_dvrs)
        }                                   # strategies
        self.d_avg = {
            "avg": self._avg,
            "macro": self._macro_avg,
            "entropy": self._entropy,
            "oracle": self._oracle
        }                                   # averages (includes oracle)
            # features
        self.Ft = Featurer()                # to add/remove model features
        
        # Private methods #
        #-----------------#
    def _iter_tok(self):
        """Iterates over the entire dataset."""
        for i, tpl in enumerate(self.files):
            nx, ny = self.files[i]
            for j, sx in enumerate(nx):
                for k, dt in enumerate(sx):
                    yield i, nx, j, k, dt, ny[j][k]
    def _make_table(self):
        """From self.files, fills a table (dictionary, not Pandas)."""
        self.table = {}
        for i, nx, j, k, dt, pos in self._iter_tok():
            tok = dt['token']+"_"+pos
            if i not in self.table:             # file to table
                self.table[i] = {'count':0, 'hist':{}}
            self.table[i]['count'] += 1         # file's token count
            if tok not in self.table[i]['hist']:# token_pos to histogram
                self.table[i]['hist'][tok] = 0
            self.table[i]['hist'][tok] += 1
        return self.table
    def _by_files(self, f, s, lim):
        """Organizes the dataset by files.
           Fills the words data."""
        self.files, on = [], ""
        if not os.path.isfile(f):       # no file to load
            return self.files
        for sequ in iter_sequ(f, s, lim):
            n = sequ[0]['file']         # current 'file'
            if on != n:                 # new 'file'
                self.files.append([[],[]]); on = n
                print(n, end=" "*40+"\r")
            for i, d in enumerate(sequ):# for each word...
                x, y = d['token'], d['pos']
            x, y = prep_sequ(sequ)
            self.files[-1][0].append(x); self.files[-1][1].append(y)
        self._make_table()
        return self.files
    def _iter_pred(self, y_tpl, ch_cutoff=True, x=[]):
        """Iterates over dicts."""
        for i, s in enumerate(y_tpl):
            for j, d in enumerate(s):
                yp, yc, yd = d['pos'], d['confidence'], d['dict']
                if ch_cutoff and yc >= self.acc:
                    continue
                if x:
                    yield x[i][j]['token'], yp, yc, yd
                else:
                    yield yp, yc, yd
    def _avg(self, acc, y_tpl, ch_cutoff=True):
        """Calculates an average (of confidence scores)."""
        y_cnf = []
        for yp, yc, yd in self._iter_pred(y_tpl, ch_cutoff=ch_cutoff):
            y_cnf.append(yc)
        return np.mean(y_cnf) if len(y_cnf) > 0 else 0.
    def _macro_avg(self, acc, y_tpl, ch_cutoff=True):
        """Calculates a macro average (of confidence scores)."""
        d_cnf = {}
        for yp, yc, yd in self._iter_pred(y_tpl, ch_cutoff=ch_cutoff):
            if yp not in d_cnf:
                d_cnf[yp] = []
            d_cnf[yp].append(yc)
        avg = [np.mean(cnf) for cnf in d_cnf.values()]
        return np.mean(avg) if avg else 0.
    def _entropy(self, acc, y_tpl, ch_cutoff=False):
        """Calculates the entropy (of confidence scores)."""
        y_cnf = []
        for yp, yc, yd in self._iter_pred(y_tpl, ch_cutoff=False):
            probs = np.array(list(yd.values()))
            y_cnf.append(entropy(probs, base=2))
        return np.mean(y_cnf) if len(y_cnf) > 0 else 0.
    def _oracle(self, acc, y_tpl, ch_cutoff=False):
        """Returns the accuracy."""
        return acc
    def _sparse_cosine(self, h1, h2):
        """Computes similarity between token histograms."""
        shrd = set(h1) & set(h2)
        dot = sum(h1[k]*h2[k] for k in shrd)
        nh1 = sum(c**2 for c in h1.values())**0.5
        nh2 = sum(c**2 for c in h2.values())**0.5
        return dot/((nh1*nh2) + 1e-6)
    def _max_sparse(self, hist, list_hist):
        """Returns the redundancy score."""
        if not list_hist:
            return 0.
        return max(self._sparse_cosine(hist, h) for s, h in list_hist)
    def _pred_dvrs(self, favg=None): ## ACTIVE LEARNING STRATEGY
        """Fills 'self.prf' for '_pick_dvrs'."""
        favg = self._avg if not favg else favg
        self.prf = [0. for i in self._ind]
        for a, i in enumerate(self._ind):          # for each file index...
            x_te, y_te = self.files[i]             # file content
            acc, y_tpl = self.pred(x_te, y_te, ch_acc=False) # prediction
            htok, tot = self.table[i]['hist'], self.table[i]['count']
            htok = {k: v/tot for k, v in htok.items()}
            gtot = sum(self.tok.values())
            ghtok = {k: v/gtot for k, v in self.tok.items()}
            wf = self._sparse_cosine(htok, ghtok)  # redundancy
            wt = favg(acc, y_tpl, ch_cutoff=True)  # token weight
            self.prf[a] = (1-wt)+(1-wf)            # formula
    def _pred_files(self, favg=None): ## ACTIVE LEARNING STRATEGY
        """Fills 'self.prf' for '_pick_file'."""
        favg = self._avg if not favg else favg
        self.prf = [0. for i in self._ind]
        for a, i in enumerate(self._ind):          # for each file index...
            x_te, y_te = self.files[i]             # file content
            acc, y_tpl = self.pred(x_te, y_te, ch_acc=False) # prediction
            wt = favg(acc, y_tpl, ch_cutoff=True)  # token weight
            self.prf[a] = (1-wt)                   # formula
    def _pred_tokens(self, x_te=[], y_te=[], favg=None): ## ACT.LEARN. STRATEGY
        """Fills 'self.prf' for '_pick_toks'."""
        if not x_te:                                # if not subset...
            x_te, y_te = self.ref                   # reference dataset
        conf, self.prf = {}, {}
        _, y_tpl = self.pred(x_te, y_te, ch_acc=False) # prediction
        for t, yp, yc in self._iter_pred(y_tpl, ch_cutoff=True, x=x_te):
            t = t+"_"+yp 
            if t not in conf:
                conf[t] = []
            conf[w].append(y_tpl[a]['confidence'])
        for w, l_cnf in conf.items():               # average all
            conf[w] = np.log10(len(l_cnf))*(1-np.mean(l_cnf))
        for a in range(10):                         # hardcoded nb_tokens
            w = max(conf, key=conf.get)
            self.prf[w] = conf.pop(w)
    
        # Data #
        #------#
    def load_dataset(self, f="ofrom_alt.joblib", s=0, lim=-1):
        """Loads "raw" data and parses it."""
        return self._by_files(f, s, lim) if f else ([], None, None)
    def load_parsed(self, f="ofrom_gen.joblib"):
        """Loads the pre-parsed data."""
        tpl = joblib.load(f)
        if isinstance(tpl, tuple):
            self.files, self.table = tpl[0], tpl[1]
        else:
            self.files = tpl; self._make_table()
        return self.files
    def save(self, f="ofrom_gen.joblib"):
        """Saves parsed data as a '.joblib' file."""
        joblib.dump((self.files, self.table), f, compress=5)
    def count(self, files=[]):
        """Returns the number of tokens."""
        n_file, n_sequ, n_tok = 0, 0, 0
        files = self.files if not files else files
        for x, y in files:
            n_file += 1
            for sequ in x:
                n_sequ += 1
                for w in sequ:
                    n_tok += 1
        return n_file, n_sequ, n_tok
    def decompose(self, wf="", nb_toks=5, lim=-1):
        """Make as many files as there are sequences.
           Cut the sequences into units of max 'nb_toks' tokens."""
        files = []
        c, lf = 0, len(self.files)
        for i in range(lf):        # for each file...
            x, y = self.files[i]
            print(f"{i+1}/{lf}", end=" "*40+"\r")
            for a, sequ in enumerate(x):        # for each sequence...
                lsequ = y[a]
                c += len(sequ)
                if nb_toks > 0:
                    while len(sequ) > nb_toks:      # max nb_toks tokens
                        s, sequ = sequ[:nb_toks], sequ[nb_toks:]
                        l, lsequ = lsequ[:nb_toks], lsequ[nb_toks:]
                        files.append([[s], [l]])
                if len(sequ) > 0:
                    files.append([[sequ], [lsequ]])
                if lim > 0 and c >= lim:        # enough tokens
                    break
            if lim > 0 and c >= lim:
                break
        self.files = files
        self._make_table()
        if wf:
            self.save(wf)
    def get_dataset(self):
        """Returns the entire dataset."""
        X, y = [], []
        for nx, ny in self.files:
            X = X+nx; y = y+ny
        return X, y
    
        # 'File' selection #
        #------------------#
    def _pick_100(self):
        """Picks index 100."""
        i = 100 if len(self._ind) > 100 else len(self._ind)-1
        v = self._ind.pop(i); return v
    def _pick_last(self):
        """Picks last."""
        v = self._ind.pop(); return v
    def _pick_rand(self):
        """Picks at random."""
        a = np.random.randint(0, len(self._ind))
        v = self._ind.pop(a); return v
    def _pick_file(self):
        """Picks based on file confidence.
           Assumes 'self.crf' is set."""
        gi, gw = -1, -1.                    # global index/weight
        for i, w in enumerate(self.prf):    # for each file index...
            if w > gw:
                gi, gw = i, w
        v = self._ind.pop(gi); self.prf.pop(gi)
        return v
    def _pick_toks(self):
        """Picks based on token confidence.
           Assumes 'self.crf' is set."""
        ga, gw = -1, -1.
        for a, i in enumerate(self._ind):   # for each file...
            lw, fh = [], self.table[i]['hist']
            for w in self.prf:              # nb_occ * avg_conf_score
                n = fh[w] if w in fh else 0
                lw.append(n*self.prf[w])
            lw = np.mean(lw)                # average
            if lw > gw:
                ga, gw = a, lw
        v = self._ind.pop(ga)
        return v
    def set_size(self, lim=-1):
        """Selects the amount of data for selection (+ reference)."""
        self._ind = [i for i in range(len(self.files))]
        if lim > 0: # remove the given amount
            nf, ns, nt = self.count()
            if (nt-lim) > nt/2:
                l_tmp, gs = [], 0
                for i, tpl in enumerate(self.files):
                    for s in tpl[0]:
                        gs += len(s)
                    l_tmp.append(i)
                    if gs >= lim:
                        break
                self._ind = l_tmp
            else:
                self.select("last", lim=(nt-lim))
        self.s = 0
    def set_ref(self, lim=100000, fixed=False):
        """Selects the reference dataset."""
        strat = "rand" if not fixed else "100"
        x, y = self.select(strat, lim=lim)
        self.ref, self.s = [x, y], 0
    def reset(self, data_size=-1, ref_size=100000, 
              fixed=False, features=False, **kwargs):
        """Resets the selection list. Gets a (100k) reference subset."""
        if features:
            self.add_ft()                               # features
        self.set_size(data_size)                        # available data
        self.set_ref(ref_size, fixed)                   # reference dataset
        self.tok = {}
    def select(self, strat="rand", X=[], y=[], lim=10000, avg="avg"):
        """Generic 'file' selection function. User can define the strategy.
           Defaults to random."""
        strat, sub = self.strat[strat]
        if sub:                                 # prepare selection
            sub(favg=self.d_avg[avg])
        old_s = self.s
        while True:
            if (not self._ind) or (lim > 0 and self.s//lim > old_s//lim):
                break
            a = strat()                         # file index
            nx, ny = self.files[a]              # file content
            tot, hist = self.table[a]['count'], self.table[a]['hist']
            self.s += tot                       # data size
            for tok, tc in hist.items():        # data histogram
                if tok not in self.tok:
                    self.tok[tok] = 0
                self.tok[tok] += 1
            X = X+nx; y = y+ny                  # add to subset
        return X, y
    
        # Training #
        #----------#
    def add_ft(self):
        """Add features to the dataset."""
        X, y = self.get_dataset()
        self.Ft.reset(); self.Ft.prepare(X, y)
        for i, tpl in enumerate(self.files):
            nx, ny = self.Ft.set(*tpl, ch_prep=False)
            self.files[i][0] = nx
        self.Ft.reset()
        self.c1, self.c2 = 0.231, 0.0004    # hyperparameters
        print("Featured.", end=" "*40+"\r")
    def rem_ft(self):
        """Remove all features but 'token'."""
        for i, tpl in enumerate(self.files):
            nx,_ = self.Ft.rem(*tpl)
            self.files[i][0] = nx
        self.c1, self.c2 = 0.0127, 0.023    # hyperparameters
    def tr(self, X, y):
        """Only updates the crf."""
        self.crf = train(X, y, self.c1, self.c2); return self.crf
    def pred(self, X, y, y_pr=None, ch_acc=True):
        """Generates an accuracy score."""
        y_tpl = [predict_one(self.crf, s) for s in X] \
                if not y_pr else y_pr
        y_ate, y_apr = [], []
        for a in range(len(y)):                     # for each sequence...
            for b in range(len(y[a])):              # for each token...
                y_ate.append(y[a][b])               # actual PoS
                y_apr.append(y_tpl[a][b]['pos'])    # predicted PoS
        acc = acc_score(y_ate, y_apr) if ch_acc else -1.
        return acc, y_tpl
    def train(self, X, y):
        """Returns an accuracy score."""
        self.tr(X, y)
        return self.pred(*self.ref)
    def train_all(self):
        """Trains a CRF on the complete dataset."""
        return self.tr(*self.get_dataset())
    def optimize(self, lim=100000, verbose=True):
        """First batch is random, next by active learning strategy."""
        from ofrom_crf import test_crf
        self.reset(ref_size=lim)
        hp = test_crf(*self.ref, verbose=verbose).best_params_
        return hp['c1'], hp['c2']

        # Iterators #
        #-----------#
    def it(self, X, y, lim, strat="rand", avg="avg"):
        """Selects and trains new files."""
        X, y = self.select(strat, X, y, lim, avg)   # select 'files'
        self.acc,_ = self.train(X, y)               # train/test
        return X, y, self.acc
    def iter(self, lim=-1, strat="rand", avg="avg", **kwargs):
        """Iterates with a given strategy."""
        self.reset(**kwargs)
        substrat = "100" if kwargs.get('fixed', False) else "rand"
        X, y, acc = self.it([], [], lim, substrat, avg)
        yield acc; del substrat
        while self._ind:                         # exhaust all files
            X, y, acc = self.it(X, y, lim, strat)
            yield acc
    def oracle(self, lim=100000, features=False, fixed=True, verbose=True):
        """Adds file by file, training the model on all files each time."""
        data_size = lim+100000 if lim > 0 else lim
        self.reset(data_size=data_size, features=features, fixed=fixed)
        X, y = [], []; Xc, yc = [], []; l_ind = []
        self.s = 0
        while self._ind:
            gacc, ga, gi = -1., -1, -1
            for a, i in enumerate(self._ind):
                nx, ny = self.files[i]
                if verbose:
                    print(f"Oracle: {len(l_ind)+1}, ({self.s} tokens)"+
                          f" {a+1}/{len(self._ind)}", 
                          end=" "*40+"\r")
                Xc, yc = X+nx, y+ny
                acc, _ = self.train(Xc, yc)
                if acc > gacc:
                    gacc, ga, gi = acc, a, i
            l_ind.append(gi); self._ind.pop(ga)
            nx, ny = self.files[gi]
            for s in nx:
                self.s = self.s + len(s)
            X, y = X+nx, y+ny
            yield gacc, l_ind



