
import numpy as np
import joblib

class Sim:
    """Procedurally-generated data of PoS annotation."""
    
    def __init__(self, toks=[], pos=[]):
        self.files = []                     # simulated data
        self.wf = ""                        # output file path
        self.p = {
            'toks': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            'pos': ['A', 'B', 'C', 'D'],
            'size': 1000000,
            'p_sh': {},
            's_sh': {}
        }                                   # parameters
        self.ld, self.lf = 0, 0             # data/file lengths in tokens
        
        # Shaping #
        #---------#
    def point_random(self):
        """Token-pos pairs at random."""
        self.p['p_sh'] = {tok:[] for tok in self.p['toks']}
    def point_determ(self):
        """Token-pos pairs deterministic (all only 1)."""
        self.p['p_sh'] = {}
        for tok in self.p['toks']:
            pos = np.random.choice(self.p['pos'])
            self.p['p_sh'][tok] = [(pos, 1.)]
    def point_probs(self):
        """Token-pos pairs probabilistic (all +2)."""
        self.p['p_sh'] = {}
        for tok in self.p['toks']:
            l_p, l_r = [], self.p['pos'].copy()
            l_pr = np.zeros(len(l_r))
            n = np.random.randint(1, len(l_r)+1)
            for a in range(n):              # add pos
                pos = l_r.pop(np.random.randint(0, len(l_r)))
                pr = np.random.uniform(0., 0.8)
                l_p.append(pos); l_pr[a] = pr
            l_pr /= l_pr.sum()              # normalize
            self.p['p_sh'][tok] = list(zip(l_p, l_pr))
    
        # Generation #
        #------------#
    def reset(self):
        """Empties properties."""
        self.files = []
    def gen_point(self):
        """Simulates a single datapoint."""
        tok = np.random.choice(self.p['toks'])
        if tok in self.p['p_sh']:           # user-defined distribution
            l_pos, l_pr = zip(*self.p['p_sh'][tok]) # assume valid input
            pos = np.random.choice(l_pos, p=l_pr)
        else:                               # default to random
            pos = np.random.choice(self.p['pos'])
        return {'token': tok}, pos
    def gen_sequ(self):
        """Simulate a sequence."""
        seq_s = int(np.random.normal(loc=6, scale=2))
        seq_s = 1 if seq_s <= 0 else seq_s
        sequ_x, sequ_y = [], []
        for _ in range(seq_s):
            x, y = self.gen_point()
            sequ_x.append(x), sequ_y.append(y)
        self.lf += seq_s
        return sequ_x, sequ_y
    def gen_file(self):
        """Simulates a file."""
        lf, self.lf = np.random.randint(200, 4001), 0
        file = [[], []]
        while self.lf < lf:
            x, y = self.gen_sequ()
            file[0].append(x); file[1].append(y)
        self.ld += self.lf
        return file
    def gen_data(self):
        """Simulates a dataset."""
        dat, self.ld = [], 0
        while self.ld < self.p['size']:
            dat.append(self.gen_file())
            print(f"Files: {self.ld}/{self.p['size']}", end="\r")
        self.files = self.files+dat
        return dat

        # File #
        #------#
    def save(self, wf=""):
        """Saves the dataset (as a joblib file)."""
        wf = wf if wf else self.wf
        print("Saving...", end=" "*40+"\r")
        joblib.dump(self.files, wf, compress=5)
        print(" "*40, end="\r")
    
if __name__ == "__main__":
    sim = Sim()
    sim.point_probs()
    sim.gen_data()
    sim.save("sim_probs.joblib")