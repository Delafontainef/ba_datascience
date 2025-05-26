from ofrom_crf import train, predict_one, acc_score
import threading as thr
import pandas as pd
import numpy as np
import os, re, time, json, joblib

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

class Gen:
    """Handles 'file' partition of the dataset and training iteration.
    
    Iterators only return accuracy. Token list and confidence dictionary
    are stored internally."""
    
    def __init__(self, f="", s=0, lim=-1, keep=[], mt=1., mf=0., mc=1.):
            # operations
        self.rng = np.random.default_rng()
        self.keep = ['ADV', 'CON', 'DET', 'PRO', 'PRP'] if not keep else keep
            # data
        self.files, self.table, self.pos, self.ref = [], {}, {}, []
        self.load_dataset(f, s, lim)        # fills the above
        self._ind = [i for i in range(len(self.files))] # file selection
        self.s = 0                          # subset size in tokens
        self.crf = None                     # CRF model
        self.prf = []                       # prediction for active learning
        self.mt, self.mf, self.mc = mt, mf, mc # weights for formula
        
        # Private methods #
        #-----------------#
    def _files_weight(self):
        """Calculates the file weight/cost."""
            # Rework the global token dictionary
        conf = {}
        for tok, set_pos in self.pos.items():
            ch_non, ch_pos = False, False
            if len(set_pos) == 1:
                ch_non = True
            for pos in set_pos:
                pos = pos.split(":", 1)[0] if ":" in pos else pos
                if pos in self.keep:
                    ch_pos = True; break
            conf[tok] = (ch_non, ch_pos)
            # Set file weights
        minc, maxc = np.inf, 0
        for i, row in self.table.iterrows(): # for each file...
            w, c, tot = 0., 0, 0
            for tok, count in row.items():   # for token type...
                if "#" in tok:
                    continue
                ch_non, ch_pos = conf[tok]; tot += count
                if not ch_non:  # only problematic tokens corrected
                    c += count
                if ch_pos:      # focus on selected 'PoS'
                    w += count
            self.table.at[i, '#weight'] = w/tot
            self.table.at[i, '#cost'] = c
            minc = c if c < minc else minc
            maxc = c if c > maxc else maxc
        self.table = self.table[['#file', '#weight', '#cost']]
        for i, row in self.table.iterrows(): # standardize cost
            self.table.at[i, '#cost'] = (c-minc)/(maxc-minc)
    def _by_files(self, f, s, lim):
        """Organizes the dataset by files.
           Fills the words data."""
        self.files, self.pos, self.ref, on = [], {}, [], ""
        self.table = {'#file':[], '#weight':[], '#cost':[]}
        if not os.path.isfile(f):       # no file to load
            return self.files, self.table, self.pos, self.ref
        for sequ in iter_sequ(f, s, lim):
            n = sequ[0]['file']         # current 'file'
            if on != n:                 # new 'file'
                self.files.append([[],[]]); on = n
                print(n, end=" "*40+"\r")
                for col in self.table:
                    self.table[col].append(0)
                self.table['#file'][-1] = n
                self.table['#weight'][-1] = None
            for i, d in enumerate(sequ):# for each word...
                x, y = d['token'], d['pos']
                if x not in self.pos:   # add to global
                    self.pos[x] = set()
                    self.table[x] = [0 for i in 
                                     range(len(self.table['#file']))]
                self.pos[x].add(y)
                self.table[x][-1] += 1
            x, y = prep_sequ(sequ)
            self.files[-1][0].append(x); self.files[-1][1].append(y)
        self.table = pd.DataFrame(self.table) # turn to DataFrame
        self._files_weight()
        return self.files, self.table, self.pos, self.ref
    def _wrap_thr(self, fun, lv, i, *args):
        """Wrapper for threads."""
        lv[i] = fun(*args)
    def _pred_files(self): ## ACTIVE LEARNING STRATEGY
        """Fills 'self.pred' for '_pick_conf'."""
        self.prf = [0. for i in self._ind]
        for a, i in enumerate(self._ind):         # for each file index...
            row = self.table.iloc[i]
            wf, cf = row['#weight'], row['#cost'] # file metadata
            x_te, y_te = self.files[i]            # file content
            _, wt = self.pred(x_te, y_te, ch_acc=False) # avg confidence score
            w = (((1-wt)*self.mt)*(wf*self.mf))/(cf*self.mc) # formula
            self.prf[a] = (a, i, w) # store
    
        # Data #
        #------#
    def load_dataset(self, f="ofrom_alt.joblib", s=0, lim=-1):
        """Loads "raw" data and parses it."""
        return self._by_files(f, s, lim) if f else ([], None, None)
    def load_parsed(self, f="ofrom_gen.joblib"):
        """Loads the pre-parsed data."""
        self.files, self.table, self.pos = joblib.load(f); self.reset()
        return self.files, self.table, self.pos
    def save(self, f="ofrom_gen.joblib"):
        """Saves parsed data as a '.joblib' file."""
        joblib.dump((self.files, self.table, self.pos), f, compress=5)
    def set_mods(self, mt=-1., mf=-1., mc=-1.):
        """Sets weights for 'pick_conf'."""
        self.mt = mt if mt >= 0. else self.mt
        self.mf = mf if mf >= 0. else self.mf
        self.mc = mc if mc >= 0. else self.mc
    def count(self):
        """Returns the number of tokens."""
        n_file, n_sequ, n_tok = 0, 0, 0
        for x, y in self.files:
            n_file += 1
            for sequ in x:
                n_sequ += 1
                for w in sequ:
                    n_tok += 1
        return n_file, n_sequ, n_tok
    
        # Training #
        #----------#
    def tr(self, X, y):
        """Only updates the crf."""
        self.crf = train(X, y); return self.crf
    def pred(self, X, y, y_pr=None, ch_acc=True):
        """Generates an accuracy score."""
        y_tpl = [predict_one(self.crf, s) for s in X] \
                if not y_pr else y_pr
        y_ate, y_apr, y_cnf = [], [], []
        for a in range(len(y)):                     # for each sequence...
            for b in range(len(y[a])):              # for each token...
                y_ate.append(y[a][b])               # actual PoS
                y_apr.append(y_tpl[a][b]['pos'])    # predicted PoS
                y_cnf.append(y_tpl[a][b]['confidence']) # confidence score
        conf = np.mean(y_cnf)
        acc = acc_score(y_ate, y_apr) if ch_acc else -1.
        return acc, conf
    def train(self, X, y):
        """Returns an accuracy score."""
        self.tr(X, y)
        return self.pred(*self.ref)
    
        # 'File' selection #
        #------------------#
    def reset(self, l_toks=[]):
        """Resets the selection list. Gets a reference subset.
           Hardcoded to 100,000 tokens."""
        self._ind = [i for i in range(len(self.files))]
        x, y = self.select(self._pick_rand, lim=100000)
        self.ref, self.s = [x, y], 0
    def _pick_rand(self):
        """Picks at random."""
        a = np.random.randint(0, len(self._ind))
        v = self._ind.pop(a); return v
    def _pick_conf(self, lim=-1):
        """Picks based on file/token.
           Assumes 'self.crf' is set."""
        ga, gi, gw = -1, -1, -1.            # global index/weight
        for a, i, w in self.prf:            # for each file index...
            if w > gw:
                ga, gi, gw = a, i, w
        v = self._ind.pop(gi); self.prf.pop(ga)
        return v
    def select(self, strat=None, X=[], y=[], lim=10000):
        """Generic 'file' selection function. User can define the strategy.
           Defaults to random."""
        nw, strat = 0, self.pick_rand if strat == None else strat
        if strat == self._pick_conf:
            self._pred_files()
        old_s = self.s
        while True:
            if (not self._ind) or (lim > 0 and self.s//lim > old_s//lim):
                return X, y                     # break
            a = strat()
            nx, ny = self.files[a]
            for s in nx:
                self.s += len(s)
            X = X+nx; y = y+ny
    def sel_rand(self, lim=10000, X=[], y=[]):
        """Selects files randomly."""
        return self.select(self._pick_rand, X, y, lim)
    def sel_conf(self, lim=10000, X=[], y=[]):
        """Selects files by confidence score."""
        return self.select(self._pick_conf, X, y, lim)
    
        # Iterators #
        #-----------#
    def it(self, X, y, lim, strat=None):
        """Selects and trains new files."""
        X, y = self.select(strat, X, y, lim)    # select 'files'
        acc,_ = self.train(X, y)                # train/test
        return X, y, acc
    def iter(self, lim=-1, strat=None, **kwargs):
        """Iterates with a given strategy."""
        self.reset(); X, y = [], []
        strat = self._pick_rand if not strat else strat
        while self._ind:                         # exhaust all files
            X, y, acc = self.it(X, y, lim, strat)
            yield acc
    def iter_passive(self, lim=-1, **kwargs):
        """Picks files at random."""
        yield from self.iter(lim, self._pick_rand)
    def iter_active(self, lim=10000, **kwargs):
        """First batch is random, next by active learning strategy."""
        self.reset()
        X, y, acc = self.it([], [], lim, self._pick_rand)
        yield acc
        while self._ind:
            X, y, acc = self.it(X, y, lim, self._pick_conf)
            yield acc

if __name__ == "__main__":
    gen = Gen()
    gen.load_dataset()
    gen.save()