import numpy as np
import scipy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn_crfsuite.estimator import CRF
from sklearn.metrics import accuracy_score as skl_acc

    # Features #
    #----------#
class Modeller:
    """Class built to generate and parse data."""
    def __init__(self):
        self.idx = "token"
        self.pos = {}
        self.cot = ['p1', 'p2', 'p3', 's1', 's2', 's3']
    
def iter_x(X):
    """Iterates over all elements in all sequences of 'X'."""
    for i, s in enumerate(X):
        for j, dw in enumerate(s):
            yield i, j, dw
def set_pos(X, y, idx="token"):
    """Generates a dictionary of token pos."""
    pos = {}
    for i, j, dw in iter_x(X):
        w, yi = dw[idx], y[i][j]
        if w not in pos:
            pos[w] = set()
        set.add(yi)
    return pos
def _acot(tok, dw, X, i, idx, w, p, ft):
    """Positional code for 'self._addcot()'."""
    pw = X[i][p][idx]
    if w in tok and pw in tok[w][ft]:
        dw[ft] = pw
    elif not w in tok:
        dw[ft] = pw
    else:
        dw[ft] = "?"
    return dw
def _addcot(tok, dw, X, i, j, w, idx, cont):
    """Positional code for 'add_ft()'."""
    mid = len(cont)//2
    for b in range(mid):                    # left cotext
        p, ft = j-(b+1), cont[b]
        if p < 0:
            dw[ft] = "?"; continue
        dw = _acot(tok, dw, X, i, idx, w, p, ft)
    for b in range(mid):                    # right cotext
        p, ft = j+(b+1), cont[mid+b]
        if p >= len(X[i]):
            dw[ft] = "?"; continue
        dw = _acot(tok, dw, X, i, idx, w, p, ft)
    return dw
def add_ft(dw, X, i, j, w, tok, pos={},
           idx="token", cont=['p1', 'p2', 'p3', 's1', 's2', 's3']):
    """Add features to a single datapoint."""
    if len(dw) > 1:                         # already 'featured'
        return dw
    w = dw[idx]
    dw['l3'] = w[-3:]
    dw['l4'] = w[-4:]
    if w in pos:
        pos = list(pos[w])
        dw['pos'] = pos[0] if len(pos) == 1 else "?"
    else:
        dw['pos'] = "?"
    dw = _addcot(tok, dw, X, i, j, w, idx, cont)
    return dw
def _scot(tok, X, i, idx, w, p, ft):
    """Positional code for '_setcot()'."""
    pw = X[i][p][idx]
    if pw in tok[w][ft]:                    # already in
        tok[w][ft][pw] += 1
    elif len(tok[w][ft]) < 100:             # limited space
        tok[w][ft][pw] = 1
    return tok
def _setcot(tok, X, i, j, w, idx, mid, cont):
    """Positional code for 'set_ft()'."""
    for b in range(mid):                    # left cotext
        p, ft = j-(b+1), cont[b]
        if p < 0:
            continue
        tok = _scot(tok, x, i, idx, w, p, ft)
    for b in range(mid):                    # right cotext
        p, ft = j+(b+1), cont[mid+b]
        if p >= len(x[i]):
            continue
        tok = _scot(tok, x, i, idx, w, p, ft)
    return tok
def set_ft(X, idx="token", cont=['p1', 'p2', 'p3', 's1', 's2', 's3']):
    """Adds model features in X."""
    tok = {}                                # positional features
    d_ind = {}; mid = len(cont)//2
    for i, j, dw in iter_x(X):              # for each element...
        w = dw[idx]
        if w not in tok:                    # add to table
            tok[w] = {k:{} for k in cont}
        tok = _setcot(tok, X, i, j, w, idx, mid, cont)
    for w in tok:                           # reduce number
        for k in cont:
            while len(tok[w][k]) > 10:
                mw = min(tok[w][k], key=tok[w][k].get)
                tok[w][k].pop(mw)
    for i, j, dw in iter_x(X):
        X[i][j] = add_ft(dw, X, i, j, w, tok, idx=idx, cont=cont)
    self.tok = {}                               # empty memory
def rem_ft(X, keep=['token']):
    """Removes all features not in 'keep'."""
    for s in X:
        for dw in s:
            dw = {k:dw[k] for k in dw if k in keep}
    return X

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
        l_res[a] = {'pos': k, 'confidence': res[k]}
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
    
    tags = sorted(set(tag for s in y for tag in s))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8)
    
    crf = CRF()
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05)
    }
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=tags)
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
              labels=tags, digits=3))
    return rs