import numpy as np
import threading as thr
from multiprocessing.pool import ThreadPool
import time

class Temp:
    def __init__(self):
        self.k = thr.Lock()
        
    def do_stuff(self, arg1, arg2):
        start, c = time.time(), 0
        while time.time()-start < np.random.randint(5, 10):
            for a in range(np.random.randint(1000000, 5000000)):
                c += 1
        return c
        
def pool_stuff():
    pool, tmp = ThreadPool(10), Temp()
    l_res = []
    for a in range(10):
        l_res.append(pool.apply_async(tmp.do_stuff,
                                      args=("hello", "world")))
        print(f"Created thread {a+1}")
    pool.close()
    print("Joining...")
    pool.join()
    print("Results:")
    for res in l_res:
        print(res.get())
        
pool_stuff()