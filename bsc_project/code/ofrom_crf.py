from sklearn.model_selection import train_test_split, cross_val_score
from sklearn_crfsuite.estimator import CRF
from sklearn.metrics import accuracy_score as skl_acc
import numpy as np

    # Features #
    #----------#
class Featurer:
    """Class built to generate and parse data."""
    
    def __init__(self, pos=None):
        self.X, self.y = [], []                         # no copy, just access
        self.idx = "token"
        self.pos = {} if not pos else pos               # PoS per token
        self.dcot = {}                                  # token neighbors
        self.cot = ['p1', 'p2', 'p3', 's1', 's2', 's3'] # nb position' features
        self.mid = len(self.cot)//2                     # before/after
        
        # Private methods #
        #-----------------#
    def _iter(self, X=[], y=[]):
        """Iterates over each element."""
        X = self.X if not X else X
        y = self.y if not y else y
        for i, sequ in enumerate(X):
            for j, dt in enumerate(sequ):
                tok = dt[self.idx]
                yield i, j, tok, dt, y[i][j]
    def _acot(self, i, p, dt, tok, ft):
        """Positional - fills 'dt'."""
        p = self.X[i][p][self.idx]
        dt[ft] = "?"
        if ((tok in self.dcot and p in self.dcot[tok][ft]) or
            (not tok in self.dcot)):
            dt[ft] = p
        return dt
    def _scot(self, i, p, dt, tok, ft):
        """Positional - fills 'self.dcot'."""
        p = self.X[i][p][self.idx]
        if p in self.dcot[tok][ft]:             # already in
            self.dcot[tok][ft][p] += 1
        elif len(self.dcot[tok][ft]) < 100:     # limited space
            self.dcot[tok][ft][p] = 1
        return dt
    def _cot(self, func, i, j, dt, tok):
        """General positional code."""
        for b in range(self.mid):               # left cotext
            p, ft = j-(b+1), self.cot[b]
            if p < 0:
                continue
            dt = func(i, p, dt, tok, ft)
        for b in range(self.mid):               # right cotext
            p, ft = j+(b+1), self.cot[self.mid+b]
            if p >= len(self.X[i]):
                continue
            dt = func(i, p, dt, tok, ft)
        return dt
    
        # Main #
        #------#
    def reset(self):
        """Clears nearly everything."""
        self.X, self.y, self.pos, self.dcot = [], [], {}, {}
    def set_args(self, idx, cot):
        """Changes instance's properties."""
        self.idx, self.cot = idx, cot
    def set_pos(self, X, y):
        """Fills the 'pos' dict."""
        for i, j, tok, dt, pos in self._iter(X, y):
            if tok not in self.pos:
                self.pos[tok] = {}
            if pos not in self.pos[tok]:
                self.pos[tok][pos] = 0
            self.pos[tok][pos] += 1
    def add(self, i, j, dt, tok):
        """Add features to a single datapoint."""
        if len(dt) > 1:                         # already 'featured'
            return dt
        dt['l3'] = tok[-3:]
        dt['l4'] = tok[-4:]
        dt['pos'] = "?"
        if tok in self.pos:                     # PoS feature
            pos = list(self.pos[tok])
            dt['pos'] = pos[0] if len(pos) == 1 else "?"
        return self._cot(self._acot, i, j, dt, tok)  # positional features
    def prepare(self, X, y):
        """Sets pos/dcot and adds features."""
        self.X, self.y = X, y
        self.set_pos(X, y)                          # set PoS dict
        for i, j, tok, dt, pos in self._iter():     # fill self.dcot
            if tok not in self.dcot:
                self.dcot[tok] = {k:{} for k in self.cot}
            self._cot(self._scot, i, j, dt, tok)
        for tok in self.dcot:                       # reduce number
            for k in self.cot:
                while len(self.dcot[tok][k]) > 10:
                    mt = min(self.dcot[tok][k], key=self.dcot[tok][k].get)
                    self.dcot[tok][k].pop(mt)
    def set(self, X, y, ch_prep=True):
        """Sets pos/dcot and adds features."""
        self.X, self.y = X, y
        self.prepare(X, y)
        for i, j, tok, dt, pos in self._iter():     # add to X
            self.X[i][j] = self.add(i, j, dt, tok)
        return self.X, self.y
    def rem(self, X=[], y=[]):
        """Sets pos/dcot and adds features."""
        X = self.X if not X else X
        y = self.y if not y else y
        for i, j, tok, dt, pos in self._iter(X, y):
            X[i][j] = {self.idx:tok}
        return X, y
    def data(self):
        """Returns the stored data."""
        return self.X, self.y

    # Deploy #
    #--------#
def train(X, y, c1=0.22, c2=0.03, max_iterations=100):
    """Trains a model using the CRF class."""
    crf = CRF(c1=c1, c2=c2, max_iterations=max_iterations); crf.fit(X, y)
    return crf
def predict_one(crf, x):
    """Predicts a single sequence, with confidence level."""
    l_res = crf.predict_marginals_single(x)
    for a, res in enumerate(l_res):
        k = max(res, key=res.get)
        l_res[a] = {'pos': k, 'confidence': res[k], 'dict': res}
    return l_res
def acc_score(y, py):
    """Simple accuracy scoring."""
    return skl_acc(y, py)

    # Testing / optimizing #
    #----------------------#
def cross_test(crf, X, y, cv=5, n_jobs=-1):
    """Tests a given 'crf' model."""
    return cross_val_score(crf, X, y, cv=cv, n_jobs=n_jobs)
def test_crf(X, y, verbose=True):
    """Test CRF hyperparameters."""
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn_crfsuite import metrics
    import scipy, time
    
    tags = sorted(set(tag for s in y for tag in s))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8)
    
    crf = CRF()
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05)
    }
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='macro', labels=tags,
                            zero_division=0)
    rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1,
                            n_jobs=-1, n_iter=50, scoring=f1_scorer)
    if verbose:
        print("Ready to test.")
    st = time.time()
    rs.fit(X_tr, y_tr)
    if verbose:
        print(f"Done: {time.time()-st}s")
        print("Best parameters:", rs.best_params_)
        print("Best CV score:", rs.best_score_)
        print(f"Model size: {rs.best_estimator_.size_/1000000:0.2f}M")
    crf = rs.best_estimator_
    y_pr = crf.predict(X_te)
    if verbose:
        print("\n", metrics.flat_classification_report(y_te, y_pr, 
              labels=tags, zero_division=0, digits=3))
    return rs