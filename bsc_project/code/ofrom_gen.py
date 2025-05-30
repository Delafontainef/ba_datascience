""" 29.05.2025
'Sim' was in development and abandoned.
      It was meant for data simulation, to test the 'Gen' class.
'Gen' contains all the relevant code.

Gen methods:
- __init__/load_dataset: load and parse the original data using 'iter_sequ'.
- load_parsed: load pre-parsed data, ready for use
- reset: called before iterating, also sets aside the reference dataset
- iter_passive/active: main method to iterate
|- select: selects the next files, up to 'lim' tokens, to add to the subset
|          uses a 'strategy' (another method like '_pick_rand')
|- tr:   trains the model on the subset
|- pred: evaluates the trained model on the reference dataset

Strategies:
- _pick_rand:    selects a file at random
- _pick_conf:    selects the file with the lowest confidence score
|- _pred_files:  sets the average confidence scores for that selection
- _pick_tok:     selects the file based on a set of tokens
|- _pred_tokens: sets those tokens and their average confidence scores

More methods:
- decompose: turns the parsed data into a sequence per file
- get_ft:    fills 'self.tok' for positional features
- add_ft:    before training, adds features to the model

Note: A lot of parameters have been hardcoded.
      1) there is no iterator for a token strategy (it's not meant to exist)
      2) there is no parameter to choose whether to add features
      3) when using a decomposed dataset, limiting its size should be done
         and then done by hand (in 'self.reset()' or during decompose).
      This is not exhaustive.
Note: CRF methods are imported from 'ofrom_crf.py'.
      This script should contain no scikit-learn operation directly.
"""

from ofrom_crf import train, predict_one, acc_score, add_ft, rem_ft
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

    # Iterator #
    #----------#
class Gen:
    """Handles 'file' partition of the dataset and training iteration.
    
    At its core, it's a list of files and a list containing their indexes.
    For iteration, that list is popped each time a file is selected.
    """
    
    def __init__(self, f="", s=0, lim=-1, keep=[], mt=1., mf=0., mc=1.):
            # operations
        self.keep = ['ADV', 'CON', 'DET', 'PRO', 'PRP'] if not keep else keep
            # data
        self.files, self.table, self.pos, self.ref = [], {}, {}, []
        self.load_dataset(f, s, lim)        # fills the above
        self._ind = [i for i in range(len(self.files))] # file selection
        self.s = 0                          # subset size in tokens
        self.crf = None                     # CRF model
        self.acc = -1.                      # model accuracy
        self.prf = []                       # prediction for active learning
        self.mt, self.mf, self.mc = mt, mf, mc # weights for formula
            # features
        self.tok = None                     # positional features
        self.cont = ['p1', 'p2', 'p3', 's1', 's2', 's3']
        
        # Private methods #
        #-----------------#
    def _wrap(func, l_res, i, args):
        """Wraps around a function, puts result in shared list."""
        l_res[i] = func(*args)
    def _ch_pos(self, tok):
        """Checks a token's state (non-problematic, grammatical)."""
        set_pos = self.pos[tok]
        ch_non, ch_pos = False, False
        if len(set_pos) == 1:
            ch_non = True
        for pos in set_pos:
            pos = pos.split(":", 1)[0] if ":" in pos else pos
            if pos in self.keep:
                ch_pos = True; break
        return ch_non, ch_pos
    def _files_weight(self):
        """Calculates the file weight/cost."""
            # Rework the global token dictionary
        conf = {}
        for tok, set_pos in self.pos.items():
            ch_non, ch_pos = self._ch_pos(tok)
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
            tot = 1 if tot == 0 else tot
            self.table.at[i, '#weight'] = w/tot
            self.table.at[i, '#cost'] = c
            minc = c if c < minc else minc
            maxc = c if c > maxc else maxc
        self.table = self.table[['#file', '#weight', '#cost']]
        self.table['#cost'] = self.table['#cost'].astype(float)
        for i, row in self.table.iterrows(): # standardize cost
            c = self.table.at[i, '#cost']
            div = 1 if (maxc-minc) == 0 else (maxc-minc)
            self.table.at[i, '#cost'] = (c-minc)/(maxc-minc)
        self.table['#cost'] = self.table['#cost'].clip(1e-6) # avoid 0.
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
    def _iter_all(self):
        """Iterates over the entire dataset."""
        for nx, ny in self.files:
            for i, sx in enumerate(nx):
                for j, dw in enumerate(sx):
                    yield nx, i, j, dw, ny[i][j]
    def _iter_tok(self, nx):
        """Iterates over tokens."""
        c = 0
        for s in nx:
            for w in s:
                w = w['token']
                yield c, w
                c += 1
    def _pred_files(self): ## ACTIVE LEARNING STRATEGY
        """Fills 'self.prf' for '_pick_conf'."""
        self.prf = [0. for i in self._ind]
        for a, i in enumerate(self._ind):          # for each file index...
            row = self.table.iloc[i]
            wf, cf = row['#weight'], row['#cost']  # file metadata
            x_te, y_te = self.files[i]             # file content
            _, y_tmp = self.pred(x_te, y_te, ch_acc=False) # confidence scores
            # y_cnf = []
            # for j, w in self._iter_tok(x_te):      # check pos/acc
                # ch_non, ch_pos = self._ch_pos(w)
                # if (not ch_pos) or (y_tmp[j] >= self.acc):
                    # continue
                # y_cnf.append(y_tmp[j])
            y_cnf = [yi for yi in y_tmp if yi < self.acc] # cutoff point
            wt = np.mean(y_cnf) if len(y_cnf) > 1 else 0. # token weight
            cf = 1. if (cf*self.mc) <= 1e-6 else (cf*self.mc) # avoid zero
            w = (((1-wt)*self.mt)+(wf*self.mf))*cf # formula
            self.prf[a] = w                        # store
    def _pred_tokens(self, x_te=[], y_te=[]): ## ACTIVE LEARNING STRATEGY
        """Fills 'self.prf' for '_pick_toks'."""
        if not x_te:                                # if not subset...
            x_te, y_te = self.ref                   # reference dataset
        conf, self.prf = {}, {}
        for s in x_te:
            y_tpl = predict_one(self.crf, s)        # labels
            for a, w in enumerate(s):
                w = w['token']
                if y_tpl[a]['confidence'] >= self.acc: # cutoff point
                    continue                        ## (note: self.pos?)
                if w not in conf:
                    conf[w] = []
                conf[w].append(y_tpl[a]['confidence'])
        for w, l_cnf in conf.items():               # average all
            conf[w] = np.log10(len(l_cnf))*(1-np.mean(l_cnf))
        for a in range(10):                         # hardcoded nb_tokens
            w = max(conf, key=conf.get)
            self.prf[w] = conf.pop(w)
        if len(self.table.columns.values) > 5:      # table has tokens
            return
        d_add = {}
        deb1, deb2 = -1, 0
        for i, file in enumerate(self.files):       # count occurrences
            deb1 += 1; deb2 = 0
            for j, w in self._iter_tok(file[0]):
                if w not in d_add:
                    d_add[w] = [0 for i in range(len(self.files))]
                d_add[w][i] += 1
        self.table = pd.concat([self.table, pd.DataFrame(d_add)], axis=1)
    
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
    def decompose(self, wf="", nb_toks=5):
        """Make as many files as there are sequences.
           Cut the sequences into units of max 'nb_toks' tokens."""
        def _tofile(files, table, s, l):
            files.append([[s], [l]])
            for k in table:
                table[k].append(0)
            return files, table
        
        files, table = [], {k:[] for k in self.table}
        for i in range(len(self.files)):        # for each file...
            x, y = self.files[i]
            print(self.table.at[i, '#file'], end=" "*40+"\r")
            for a, sequ in enumerate(x):        # for each sequence...
                lsequ = y[a]
                while len(sequ) > nb_toks:      # no more than nb_toks tokens
                    s, sequ = sequ[:nb_toks], sequ[nb_toks:]
                    l, lsequ = lsequ[:nb_toks], lsequ[nb_toks:]
                    files, table = _tofile(files, table, s, l)
                if len(sequ) > 0:
                    files, table = _tofile(files, table, sequ, lsequ)
        for i in table['#file']:
            table['#file'][i] = f"f{i}"
            table['#weight'][i] = None
        self.files = files
        self.table = pd.DataFrame(table)
        self._files_weight()
        if wf:
            self.save(wf)
    def add_ft(self):
        """How about some madness?"""
        def _cot(x, i, w, p, ft, tok):
            """Positional code for 'self._ft()'."""
            pw = x[i][p]['token']
            if pw in tok[w][ft]:               # already in
                tok[w][ft][pw] += 1
            elif len(tok[w][ft]) < 100:        # limited space
                tok[w][ft][pw] = 1
            return tok
        
        tok = {}   # features
        d_ind = {}; mid = len(self.cont)//2
        for x, i, j, dw, pos in self._iter_all():   # for each token
            w = dw['token']
            if w not in tok:                        # add to table
                tok[w] = {k:{} for k in self.cont}
            for b in range(mid):                    # left cotext
                p, ft = j-(b+1), self.cont[b]
                if p < 0:
                    continue
                tok = _cot(x, i, w, p, ft, tok)
            for b in range(mid):                    # right cotext
                p, ft = j+(b+1), self.cont[mid+b]
                if p >= len(x[i]):
                    continue
                tok = _cot(x, i, w, p, ft, tok)
        for w in tok:
            for k in self.cont:
                while len(tok[w][k]) > 10:
                    mw = min(tok[w][k], key=tok[w][k].get)
                    tok[w][k].pop(mw)
        for x, i, j, dw, y in self._iter_all():
            x[i][j] = add_ft(dw, x, i, j, w, tok, pos=self.pos,
                             idx="token", cont=self.cont)
    def rem_ft(self):
        """Remove all features but 'token'."""
        for i, tpl in enumerate(self.files):
            self.files[i][0] = rem_ft(tpl[0], ['token'])
    
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
        acc = acc_score(y_ate, y_apr) if ch_acc else -1.
        return acc, y_cnf
    def train(self, X, y):
        """Returns an accuracy score."""
        self.tr(X, y)
        return self.pred(*self.ref)
    def optimize(self, lim=100000):
        """First batch is random, next by active learning strategy."""
        from ofrom_crf import test_crf
        self.reset()
        X, y = self.select(self._pick_rand, lim=lim)
        test_crf(X, y)
    
        # 'File' selection #
        #------------------#
    def reset(self, l_toks=[]):
        """Resets the selection list. Gets a reference subset.
           Hardcoded to 100,000 tokens."""
        # if len(self.files) > 50000: # FAKE code
            # self.files = self.files[:500001]
        # self.add_ft()
        # print("Featured.", end=" "*40+"\r")
        self._ind = [i for i in range(len(self.files))]
        x, y = self.select(self._pick_rand, lim=100000)
        self.ref, self.s = [x, y], 0
    def _pick_rand(self):
        """Picks at random."""
        a = np.random.randint(0, len(self._ind))
        v = self._ind.pop(a); return v
    def _pick_conf(self):
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
            lw, row = [], self.table.iloc[i]
            cf = row['#cost']
            for w in self.prf:              # nb_occ * avg_conf_score
                lw.append(row[w]*self.prf[w])
            lw = np.mean(lw) # / cf         # average
            if lw > gw:
                ga, gw = a, lw
        v = self._ind.pop(ga)
        return v
    def select(self, strat=None, X=[], y=[], lim=10000):
        """Generic 'file' selection function. User can define the strategy.
           Defaults to random."""
        nw, strat = 0, self.pick_rand if strat == None else strat
        if strat == self._pick_conf:
            self._pred_files()
        elif strat == self._pick_toks:
            self._pred_tokens()
        old_s = self.s
        while True:
            if (not self._ind) or (lim > 0 and self.s//lim > old_s//lim):
                return X, y                     # break
            a = strat()
            nx, ny = self.files[a]
            for s in nx:
                self.s += len(s)
            X = X+nx; y = y+ny
    
        # Iterators #
        #-----------#
    def it(self, X, y, lim, strat=None):
        """Selects and trains new files."""
        X, y = self.select(strat, X, y, lim)    # select 'files'
        self.acc,_ = self.train(X, y)           # train/test
        return X, y, self.acc
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
            X, y, acc = self.it(X, y, lim, self._pick_conf) # self._pick_toks)
            yield acc

    # Simulation #
    #------------#
class Sim:
    """Simulates data to test 'Gen'."""
    def __init__(self):
        self.fixed = {
            'a':'A', 'b':'A',
            'c':'B', 'd':'B',
            'e':'C', 'f':'C',
            'g':'D', 'h':'D',
            'i':'A', 'j':'B'
        }
        self.voc = list(self.fixed.keys())
        self.dat = []
        
        # Tokens #
        #--------#
    def pos_dep(self, w, pos):
        """Predictable data by position."""
        if w in ['a', 'b'] and pos == 0:
            return 'A'
        elif w in ['c', 'd'] and pos == 1:
            return 'B'
        elif w in ['e', 'f'] and pos == 2:
            return 'C'
        elif w in ['g', 'h'] and pos == 3:
            return 'D'
        elif w in ['i', 'j'] and pos == 4:
            return 'A'
        else:
            return self.fixed[w]
    def unpred(self):
        """Unpredictable label."""
        return np.random.choice(['A', 'B', 'C', 'D'], p=[.35, .25, .25, .15])
    
        # Structures #
        #------------#
    def sequ(self, typ="simple"):
        """Generates a fake sequence."""
        l_w, l_p = [np.random.choice(self.voc) for _ in range(5)], []
        for i, w in enumerate(l_w):
            r = np.random.random()
            if typ == "simple":
                if r < 0.15:
                    l = self.pos_dep(w, i)
                else:
                    l = self.fixed[w]
            elif r < 0.7:
                l = self.fixed[w]
            elif r < 0.85:
                l = self.pos_dep(w, i)
            else:
                l = self.unpred()
            l_p.append(l)
        return l_w, l_p
    def file(self, n_sequ=200, typ="simple"):
        """Generates a fake file."""
        r = np.random.randint(200, 2801)
        f, c = [[], []], 0
        for a in range(n_sequ):
            l_w, l_p = self.sequ(typ)
            f[0].append(l_w); f[1].append(l_w); c = c+5
        return f, c
    def data(self):
        """Generates a fake dataset."""
        self.dat, gn = [], 0
        while gn < 2000000:
            n_sequ = np.random.randint(200, 800)
            typ = "simple" if np.random.random() <= 0.7 else "complex"
            file, c = self.file(n_sequ, typ)
            self.dat.append(file); gn += c
        return self.dat
        
        # joblib #
        #--------#
    def tobase(self, wf="alt.joblib"):
        """Save as joblib file."""
        l_dat = []
        for a, file in enumerate(self.dat):
            fname = f"a{a}"
            for b, sequ in enumerate(file[0]):
                for c, w in enumerate(sequ):
                    l_dat.append(
                        {'file':fname, 'speaker':'spk', 'start':0., 'end':0.,
                         'token':w, 'pos':file[1][b][c]}
                    )
            l_dat.append(
                {'file':fname, 'speaker':'spk', 'start':0., 'end':10.,
                 'token':'_', 'pos':file[1][b][c]}
            )
        joblib.dump(l_dat, wf, compress=5)

if __name__ == "__main__":
    sim = Sim(); sim.data(); sim.tobase()