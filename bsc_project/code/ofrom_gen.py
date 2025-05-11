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
    are stored in properties."""
    
    def __init__(self, f="", s=0, lim=-1, keep=[], mt=0.8, mf=0.2):
        self.files, self.table = [], {}
            # operations
        self.rng = np.random.default_rng()
        self.conf = {}              # all tokens for current loop
        self.toks = []              # token selection for active training
        self.keep = ['ADV', 'CON', 'DET', 'PRO', 'PRP'] if not keep else keep
        self.mt, self.mf = mt, mf
            # data
        self.files, self.table = self.load_dataset(f, s, lim)
        self.ind = [i for i in range(len(self.files))]
            # multithreading
        self.k_conf = thr.Lock()    # lock for word dictionary
        
        # Private methods #
        #-----------------#
    def _files_weight(self):      ## ACTIVE LEARNING STRATEGY
        """Calculates the file weight/cost."""
            # Rework the global token dictionary
        ch_non, ch_pos = False, False
        for tok, set_pos in self.conf.items():
            if len(set_pos) == 1:
                ch_non = True
            for pos in set_pos:
                pos = pos.split(":", 1)[0] if ":" in pos else pos
                if pos in self.keep:
                    ch_pos = True; break
            self.conf[tok] = (ch_non, ch_pos)
            # Set file weights
        for i, row in self.table.iterrows(): # for each file...
            w, c, tot = 0., 0, 0
            for tok, count in row['#weight'].items():   # for token type...
                ch_non, ch_pos = self.conf[tok]; tot += count
                if not ch_non:  # only problematic tokens corrected
                    c += count
                if ch_pos:      # focus on selected 'PoS'
                    w += count
                tot += count
            row['#weight'] = w/tot
            row['#cost'] = c
        self.conf = {}
    def _by_files(self, f, s, lim):
        """Organizes the dataset by files.
           Fills the words data."""
        on, files = "", []
        table, tloc = {'#file':[], '#weight':[], '#cost':[]}, {}
        self.conf, tmp = {}, {}
        if not os.path.isfile(f):       # no file to load
            return files, table
        for sequ in iter_sequ(f, s, lim):
            n = sequ[0]['file']         # current 'file'
            if on != n:                 # new 'file'
                files.append([[],[]]); on = n
                print(n, end=" "*40+"\r")
                table['#file'].append(n)
                if tmp:
                    table['#weight'][-1] = tmp.copy()
                table['#weight'].append(None); tmp =  {}
                table['#cost'].append(0)
            for i, d in enumerate(sequ):# for each word...
                x, y = d['token'], d['pos']
                if x not in tmp:        # add to local
                    tmp[x] = 0
                tmp[x] += 1
                if x not in self.conf:  # add to global
                    self.conf[x] = set()
                self.conf[x].add(y)
            x, y = prep_sequ(sequ)
            files[-1][0].append(x); files[-1][1].append(y)
        if tmp:
            table['#weight'][-1] = tmp
        self.table = pd.DataFrame(table) # turn to DataFrame
        self._files_weight()
        return files, self.table
    def _tok_weight(self, i):     ## ACTIVE LEARNING STRATEGY
        """Calculates the token weight for file at index 'i'."""
        wt, wf, row = 0., 0., self.table.iloc[i]
        for tok, conf in self.toks:
            wt += row[tok]*(1.-conf)
        return wt/len(self.toks)
    def _sel_toks(self, nb_toks): ## ACTIVE LEARNING STRATEGY
        """Selects next tokens for 'file' selection."""
        l_toks = []
        for tok, tpl in self.conf.items():
            w = (1.-tpl[0])*tpl[1]     # confidence_score * nb_occurrences
            if len(l_toks) < nb_toks:
                l_toks.append((tok, tpl[0], w)); continue
            for i, ntpl in enumerate(l_toks):
                if w >= ntpl[2] or i-1 < 0:
                    continue
                l_toks[i-1] = (tok, tpl[0], w); break
        return 
    def _up_gtoks(self, ch_fixed, g_toks):
        """Updates 'g_toks' with new confidence scores."""
        if ch_fixed and g_toks:                 # keep g_toks
            for a, tpl in enumerate(g_toks):    # update 'g_toks'
                conf = self.conf[tpl[0]][0] if tpl[0] in self.conf else tpl[1]
                # conf = np.mean(self.conf[tpl[0]]) if tpl[0] in self.conf \
                       # else tpl[1]
                g_toks[a] = (tpl[0], conf)
            self.toks = g_toks
        else:                                   # replace with l_toks
            g_toks = self.toks
        return g_toks
    def _wrap_thr(self, fun, lv, i, *args):
        """Wrapper for threads."""
        lv[i] = fun(*args)

        # Data #
        #------#
    def load_dataset(self, f="ofrom_alt.joblib", s=0, lim=-1):
        """Loads "raw" data and parses it."""
        self.files, self.table = self._by_files(f, s, lim) if f else ([], None)
        self.reset()
        return self.files, self.table
    def load_parsed(self, f="ofrom_gen.joblib"):
        """Loads the pre-parsed data."""
        self.files, self.table = joblib.load(f); self.reset()
        return self.files, self.table
    def save(self):
        """Saves parsed data as a '.joblib' file."""
        joblib.dump((self.files, self.table), "ofrom_gen.joblib", compress=5)
    def get_toks(self):
        """Wrapper for token list."""
        return self.toks
    def get_conf(self):
        """Wrapper for confidence dictionary."""
        return self.conf
    def get_table(self):
        """Wrapper for file table."""
        return self.table
    def set_mw(self, mt=0.8, mf=0.2):
        """Sets weights for 'pick_conf'."""
        self.mt, self.mf = mt, mf
    
        # Training #
        #----------#
    def cross(self, X, nb_batch):
        """Cuts the sequences into 'nb' random batches."""
        lx = len(X); rx, l_ind = int(lx/nb_batch), [a for a in range(lx)]
        l_sub = [[] for a in range(nb_batch)]
        l_ind = self.rng.choice(l_ind, len(l_ind), replace=False)
        for a in range(len(l_sub)):
            l_sub[a] = l_ind[a*rx:(a+1)*rx]
        l_sub[nb_batch-1] = l_ind[(nb_batch-1)*rx:]
        return l_sub
    def tr(self, X_tr, y_tr):
        """Trains a given batch."""
        return train(X_tr, y_tr)
    def pred(self, crf, X_te, y_te, y_pr=None):
        """Generates an accuracy score."""
        y_pr = [predict_one(crf, s) for s in X_te] if not y_pr else y_pr
        y_ate, y_apr = [], []
        for a in range(len(y_te)):
            for b in range(len(y_te[a])):
                y_ate.append(y_te[a][b])
                y_apr.append(y_pr[a][b]['pos'])
        return acc_score(y_ate, y_apr)
    def upconf(self, X_te, y_pr):
        """Updates the token confidence dictionary."""
        self.k_conf.acquire()
        for a, sequ in enumerate(y_pr):
            for b, dw in enumerate(sequ):
                tok = X_te[a][b]['token']
                if tok not in self.conf:
                    self.conf[tok] = (dw['confidence'], 1)
                    # self.conf[tok] = [dw['confidence']]
                else:
                    avg, c = self.conf[tok]; c += 1
                    self.conf[tok] = ((avg+dw['confidence'])/c, c)
                    # self.conf[tok].append(dw['confidence'])
        self.k_conf.release()
    def train_batch(self, X, y, l_ind, ch_conf=True):
        """Splits, trains and returns accuracy."""
        X_tr, X_te, y_tr, y_te = X.copy(), [], y.copy(), []
        l_ind.sort(); l_ind = l_ind[::-1]
        for a in l_ind:
            X_te.append(X[a]); y_te.append(y[a]); X_tr.pop(a); y_tr.pop(a)
        crf = self.tr(X_tr, y_tr)
        y_pr = [predict_one(crf, s) for s in X_te]
        acc = self.pred(crf, X_te, y_te, y_pr)
        if ch_conf:
            self.upconf(X_te, y_pr)
        return acc
    def train(self, X, y, nb_batch=5, ch_conf=True):
        """Five-fold training/validation to cover the entire data.
           Returns the accuracy score (average)."""
        l_sub = self.cross(X, nb_batch)
        l_thr = [a for a in range(nb_batch)]
        l_sc = [a for a in range(nb_batch)]
        for a in range(len(l_sub)):                         # multicore
            l_thr[a] = thr.Thread(target=self._wrap_thr,
                       args=(self.train_batch, l_sc, a, 
                             X, y, l_sub[a], ch_conf))
            l_thr[a].start()
        for th in l_thr:                                    # wait for threads
            th.join()
        return np.mean(l_sc)
    def crf_passive(self, ch_acc=False):
        """Trains the entire dataset passively."""
        X, y = [], []
        for file in self.files:     # get all sequences
            for nx, ny in file:
                X.append(nx); y.append(ny)
        crf = self.tr(X, y)         # train (no train-test-split)
        if not ch_acc:              # just the model
            return crf
        return crf, self.train(X, y)
    
        # 'File' selection #
        #------------------#
    def reset(self, l_toks=[]):
        """Resets the selection list."""
        self.ind = [i for i in range(len(self.files))]
        self.conf, self.toks = {}, l_toks if isinstance(l_toks, list) else []
    def pick_rand(self, l_ind):
        """Picks at random."""
        if not l_ind:
            return None
        a = np.random.randint(0, len(l_ind))
        v = l_ind.pop(a); return v
    def pick_conf(self, l_ind): ## ACTIVE LEARNING STRATEGY
        """Picks based on file/token."""
        gi, gw = -1, -1.                        # global index/weight
        for i in l_ind:                         # for each file index...
            wt, row = self._tok_weight(i), self.table.iloc[i]
            wf, cf = row['#weight'], row['#cost']
            w = ((wt*self.mt)*(wf*self.mf))/cf  # formula
            if w > gw:
                gi = i; gw = w
        v = l_ind.pop(gi); return v
    def select(self, strat=None, X=[], y=[], lim=10000):
        """Generic 'file' selection function. User can define the strategy.
           Defaults to random."""
        nw, strat = 0, self.pick_rand if strat == None else strat
        while True:
            if (not self.ind) or (lim > 0 and nw >= lim):
                return X, y                     # break
            a = strat(self.ind)
            nx, ny = self.files[a]
            nw += len(nx); X = X+nx; y = y+ny
        return X, y
    def sel_rand(self, lim=10000, X=[], y=[]):
        """Selects files randomly."""
        return select(self.pick_rand, X, y, lim)
    def sel_conf(self, lim=10000, X=[], y=[]):
        """Selects files by token/file weights."""
        return select(self.pick_conf, X, y, lim)
    
        # Iterators #
        #-----------#
    def it(self, X, y, lim, nb_toks, nb_batch=5, strat=None):
        """Selects and trains new files."""
        X, y = self.select(strat, X, y, lim)    # select 'files'
        acc = self.train(X, y, nb_batch)        # train/test
        self._sel_toks(nb_toks)                 # select next tokens
        return X, y, acc
    def iter_passive(self, lim=-1, nb_batch=5, **kwargs):
        """Picks files at random."""
        self.reset(); X, y = [], []
        while self.ind:                         # exhaust all files
            X, y, acc = self.it(X, y, lim, 0, nb_batch)
            yield acc
    def iter_active(self, lim=10000, nb_toks=10, nb_batch=5,
                    ch_fixed=False, g_toks=[], **kwargs):
        """First batch is random, next batches are decided by 
           active learning stragegy."""
        self.reset(g_toks)
        if (not ch_fixed) or (not g_toks):      # variable tokens
            X, y, acc = self._add_sequs([], [], lim, nb_toks)
            g_toks = self._up_gtoks(ch_fixed, g_toks)
        else:                                   # fixed tokens
            X, y, acc = self.it([], [], lim, nb_toks, nb_batch, 
                                self._conf_pick)
            g_toks = self._up_gtoks(ch_fixed, g_toks)
        yield acc
        while self.ind:                         # exhaust all files
            X, y, acc = self.it(X, y, lim, nb_toks, nb_batch, self._conf_pick)
            g_toks = self._up_gtok(ch_fixed, g_toks,)
            yield acc

if __name__ == "__main__":
    gen = Gen()
    gen.load_dataset()
    gen.save()