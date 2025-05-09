from ofrom_crf import train, predict_one, acc_score
from multiprocessing.pool import ThreadPool
import threading as thr
import pandas as pd
import numpy as np
import os, re, time, json, joblib, zipfile

    # From 'ofrom_pos.py' #
    #---------------------#
def read(f):
    """Reads the database."""
    if not os.path.isfile(f):                   # no file
        return pd.DataFrame()
    f = os.path.abspath(f)
    d, file = os.path.split(f); fi, ext = os.path.splitext(file)
    if ext.lower() == ".zip":                   # unzip
        with zipfile.ZipFile(f, 'r') as zf:
            l_files = zf.namelist()
            for nf in l_files:                  # pick first match in zip
                nd, nfile = os.path.split(nf)
                nfi, ne = os.path.splitext(nfile)
                if fi == nfi:
                    f = os.path.join(d, nfile); break
            zf.extract(nfile, d)
            file, ext = nfile, ne
    d_e = {
        '.joblib': joblib.load, '.xlsx': pd.read_excel, '.csv': pd.read_csv,
        '.json': json.load
    }
    if ext.lower() in d_e:                      # based on extension
        return d_e[ext.lower()](f)
    return pd.DataFrame()                       # give up
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

class Ofrom_gen:
    """Divides the dataset (list of sequences) by file and keeps track 
       of the number of occurrences for each token, per file.
       
       This allows for the selection of files based on token confidence.
       
       Iterators accrue sequences and train a model on each batch,
       then return an F1 score and the (by default) 10 worst-faring tokens.
       
       The class does not contain any method for visualization."""
    
    def __init__(self, f="ofrom_alt.joblib", s=0, lim=-1):
        self.files, self.table = self.load_dataset(f, s, lim)
        self.ind = [i for i in range(len(self.files))]
            # multithreading
        self.pool = None            # pool of thread workers
        self.pool_res = []          # list for thread results
        self.k_conf = thr.Lock()    # lock for word dictionary
    
        # Private utilities #
        #-------------------#
    def _by_files(self, f, s, lim):
        """Organizes the dataset by files.
           Fills the words data."""
        on, files, table = "", [], {'#file':[]}
        lk = ['token', 'pos', 'lemma']  # only keep that data
        for sequ in iter_sequ(f, s, lim):
            if not sequ:                # unneeded safety
                continue
            n = sequ[0]['file']         # current file
            if on != n:                 # new file
                files.append([]); on = n
                table['#file'].append(n)
                for w in table:
                    if w == '#file':
                        continue
                    table[w].append(0)
            for i, d in enumerate(sequ):# for each word...
                w = d['token']
                if w not in table:      # add to 'table' dict
                    table[w] = [0 for a in range(len(files))]
                table[w][-1] += 1       # incr' word for that file
                sequ[i] = {k: d[k] for k in lk} # lighter sequ data
            files[-1].append(sequ)
        table = pd.DataFrame(table)     # turn to DataFrame
        table.set_index('#file')
        return files, table
    def _rand_pick(self, l, l_toks=None):
        """Picks at random from a list, removes that value from the list."""
        if not l:
            return None
        a = np.random.randint(0, len(l))
        v = l.pop(a); return v
    def _conf_pick(self, l, l_toks):
        """Picks based on confidence weight."""
        res, gw = -1, -1.
        for a, i in enumerate(l):
            row, w = self.table.iloc[i], 0
            for tok, conf in l_toks:    # actual weighing formula
                w += row[tok]*(1-conf)  # nb_occurrences * 1-confidence_score
            if w > gw:                  # if higher weight, select file
                res = a; gw = w
        v = l.pop(res); return v
    def _cross_tr(self, sequs, nb):
        """Cuts the sequences into 'nb' random batches."""
        lx = len(sequs); rx, l_ind = int(lx/nb), [a for a in range(lx)]
        l_sub, rng = [[] for a in range(nb)], np.random.default_rng()
        l_ind = rng.choice(l_ind, len(l_ind), replace=False)
        for a, sub in enumerate(l_sub):
            l_sub[a] = l_ind[a*rx:(a+1)*rx]
        l_sub[nb-1] = l_ind[(nb-1)*rx:]
        return l_sub
    def _tr(self, X, y, l_ind, d_conf):
        """Trains a given batch."""
        X_tr, X_te, y_tr, y_te = X.copy(), [], y.copy(), []
        l_ind.sort(); l_ind = l_ind[::-1]
            # split, train and predict
        for a in l_ind:
            X_te.append(X[a]); y_te.append(y[a]); X_tr.pop(a); y_tr.pop(a)
        crf = train(X_tr, y_tr)
        y_pr = [predict_one(crf, s) for s in X_te]
            # accuracy score
        y_ate, y_apr = [], []
        for a in range(len(y_te)):
            for b in range(len(y_te[a])):
                y_ate.append(y_te[a][b])
                y_apr.append(y_pr[a][b]['pos'])
        score = acc_score(y_ate, y_apr)
            # confidence scores
        with self.k_conf:
            for a, sequ in enumerate(y_pr):
                for b, dw in enumerate(sequ):
                    tok = X_te[a][b]['token']
                    if tok not in d_conf:
                        # d_conf[tok] = (dw['confidence'], 1)
                        d_conf[tok] = [dw['confidence']]
                    else:
                        # avg, c = d_conf[tok]; c += 1
                        # d_conf[tok] = ((avg+dw['confidence'])/c, c)
                        d_conf[tok].append(dw['confidence'])
        return score, d_conf
    def _getconf(self, d_conf, nb=10):
        """Returns 'nb' tokens with the worst confidence score."""
        l_res, tmp = [], d_conf.copy()
        for a in range(nb):
            if len(tmp) < 1:
                break
            tok = min(tmp, key=lambda s: np.mean(tmp[s]) if tmp[s] else 2.)
            l_res.append((tok, np.mean(tmp[tok]) if tmp[tok] else 0.))
            tmp.pop(tok)
        return l_res
    def _up_gtok(self, ch_fixed, l_toks, g_toks, tok_conf):
        """Updates 'g_toks' with new confidence scores."""
        if ch_fixed:    # we want to track a fixed set of tokens
            for a, tpl in enumerate(g_toks): # update 'g_toks'
                conf = tok_conf[tpl[0]][0] if tpl[0] in tok_conf \
                       else tpl[1]
                g_toks[a] = (tpl[0], conf)
            l_toks = g_toks
        else:
            g_toks = l_toks
        return l_toks, g_toks
    def _add_sequs(self, l_sequs, tok_conf, lim, nb, l_toks=None, strat=None):
        """Operations for each iteration in training."""
        start = time.time()
        l_sequs = l_sequs + self.select(strat, lim, l_toks)
        for tok in tok_conf:
            tok_conf[tok] = []
        acc_score, tok_conf = self.train(l_sequs, tok_conf)
        l_toks = self._getconf(tok_conf, nb)
        end = time.time()-start
        return l_sequs, tok_conf, acc_score, l_toks
    
        # Multithreading #
        #----------------#
    def pool_set(self, nb):
        """Sets the number of workers."""
        self.pool, self.pool_res = ThreadPool(nb), []
        return self.pool
    def pool_rem(self):
        """Just to make sure that 'close()' is followed."""
        self.pool, self.pool_res = None, []
    def pool_add(self, fun, args):
        """Wrapper for 'apply_async'."""
        self.pool_res.append(self.pool.apply_async(fun, args=args))
        return self.pool_res[-1]
    def pool_close(self):
        """If 'self.pool_read()' isn't used."""
        self.pool.close()
    def pool_read(self, ch_ind=True):
        """Generator for results."""
        try:                    # closed already?
            self.pool.close()
        except Exception:
            pass
        for i, res in enumerate(self.pool_res):
            res = res.get()
            yield (i, res) if ch_ind else res
        self.pool_rem()         # reset everything
    
        # Dataset #
        #---------#
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
    
        # Training #
        #----------#
    def train(self, sequs, d_conf={}, nb=5):
        """Five-fold training/validation to cover the entire data.
           Returns an F1 score and confidence scores per word (averages)."""
        l_sub = self._cross_tr(sequs, nb)
        X, y, gsc = [], [], 0.
        for s in sequs:                                      # fill X/y
            nx, ny = prep_sequ(s); X.append(nx); y.append(ny)
        self.pool_set(nb)
        for a in range(len(l_sub)):                          # multicore
            self.pool_add(self._tr, (X, y, l_sub[a], d_conf))
        for acc_score, conf in self.pool_read(False):
            gsc += acc_score
        return gsc/nb, d_conf
    def crf_passive(self, ch_conf=True):
        """Trains the entire dataset passively."""
        X, y, l_sequs = [], [], []
        for file in self.files:     # get all sequences
            for sequ in file:
                nx, ny = prep_sequ(sequ); X.append(nx); y.append(ny)
                if ch_conf:
                    l_sequs.append(sequ)
        crf = train(X, y)           # train (no train-test-split)
        if not ch_conf:             # just the model
            return crf, -1., {}
        return crf, self.train(l_sequs)
        
        # "File" selection #
        #------------------#
    def reset(self):
        """Resets the selection list."""
        self.ind = [i for i in range(len(self.files))]
    def select(self, strat=None, lim_w=10000, l_toks=None):
        """Generic 'file' selection function. User can define the strategy.
           Defaults to random."""
        nw, l_sequs = 0, []
        strat = self._rand_pick if strat == None else strat
        while True:
            if (not self.ind) or (lim_w > 0 and nw >= lim_w):
                return l_sequs                  # break
            a = strat(self.ind, l_toks)
            file = self.files[a]
            for sequ in file:                   # increment
                nw += len(sequ)
            l_sequs = l_sequs+file              # add file sequences
    def sel_rand(self, lim_w=10000):
        """Random 'file' selection."""
        return self.select(self._rand_pick, lim_w)
    def sel_conf(self, lim_w=10000, l_toks=None):
        """'file' selection by confidence score weight."""
        return select(self._conf_pick, lim_w, l_toks)
    
        # Iterators #
        #-----------#
    def iter_passive(self, lim=-1, nb=10):
        """Just train on all files blindly."""
        l_sequs, tok_conf = [], {}
        while self.ind:     # exhaust all files
            l_sequs, tok_conf, acc, l_toks = self._add_sequs(l_sequs, tok_conf, 
                                                             lim, 0)
            yield acc, l_toks
    def iter_active(self, lim=10000, nb=10, ch_fixed=False, g_toks=[]):
        """First batch is random, next batches are decided by 
           lowest confidence score words."""
        if (not ch_fixed) or (not g_toks):  # no pre-determined word list
            l_sequs, tok_conf, acc, l_toks = self._add_sequs([], {}, lim, nb)
            l_toks, g_toks = self._up_gtok(False, l_toks, g_toks, tok_conf)
        else:
            l_sequs, tok_conf, acc, l_toks = self._add_sequs([], {}, lim, nb,
                                              g_toks, self._conf_pick)
            l_toks, g_toks = self._up_gtok(ch_fixed, l_toks, g_toks, tok_conf)
        yield acc, l_toks
        while self.ind:                     # exhaust available data
            l_sequs, tok_conf, acc, l_toks = self._add_sequs(l_sequs, tok_conf, 
                                              lim, nb, l_toks, self._conf_pick)
            l_toks, g_toks = self._up_gtok(ch_fixed, l_toks, g_toks, tok_conf)
            yield acc, l_toks