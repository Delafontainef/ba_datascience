import numpy as np
import scipy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn_crfsuite.estimator import CRF
from sklearn.metrics import accuracy_score as skl_acc

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

def test_crf(dat_f, tag_f):
    """Test CRF hyperparameters."""
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn_crfsuite import metrics
    
    X, y = data_load(dat_f); tags = get_tags(tag_f, y)
    data_save(dat_f, X, y); save_tags(tag_f, tags)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8)
    
    crf = CRF()
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05)
    }
    tags.remove("_"); tags.remove("?")
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=tags)
    rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1,
                            n_jobs=-1, n_iter=50, scoring=f1_scorer)
    print("Ready to test.")
    st = time.time()
    rs.fit(X_tr, y_tr)
    print(f"Done: {time.time()-st}s")
    print("Best parameters:", rs.best_params_)
    print("Best CV score:", rs.best_score_)
    print(f"Model size: {rs.best_estimator_.size_/1000000:0.2f}M") 
    
    crf = rs.best_estimator_
    y_pr = crf.predict(X_te)
    print("\n", metrics.flat_classification_report(y_te, y_pr, labels=tags,
                digits=3))
