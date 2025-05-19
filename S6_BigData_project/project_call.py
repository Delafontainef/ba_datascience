import requests as r
import threading as thr
import sys, time, random
url = "http://localhost:7777"

def _r(json, rs):
    with rs.post(url, json=json) as res:
        return res.status_code, res.text
def requ(json={'size': 10, 'cycle': 1}, verbose=False,
         rs=None):
    """Sends request to server."""
    if rs is None:
        with r.Session() as rs:
            code, txt = _r(json, rs)
    else:
        code, txt = _r(json, rs)
    if verbose:
        print(code, txt)
    return (code, txt)
def _clean(l_thr):
    """Cleans the thread list."""
    return [th for th in l_thr if th.is_alive()]
def _strat(state, ch):
    """Changes start on demand.
       size: low<100, mid<10000, high>10000.
       cycle: low<100, mid<500, high>500."""
    if ch:
        state = (state+1)%5
    io = False
    if state == 0: # low, low
        s = random.randint(10, 100)
        c = random.randint(10, 100)
    if state == 1: # mid/high, low
        s = random.randint(1000, 20001)
        c = random.randint(1, 51)
        c = -1 if c > 5 else c
    elif state == 2: # low, mid/high
        s = random.randint(10, 100)
        c = random.randint(100, 2000)
    elif state == 3: # mid, mid
        s = random.randint(100, 1000)
        c = random.randint(100, 200)
    else:           # io
        s = random.randint(20, 201)
        c = random.randint(2, 20)
        io = True if random.randint(0, 101) < 5 else False
    return state, {'size':s, 'cycle':c, 'io':str(io).lower()}

def loop(t=60, rate=5., lim=100, switch=60, verbose=True):
    """Keeps sending requests: tries to avoid overloading."""
    start, end, l_thr, state, ch = time.time(), 0., [], 0, False
    mid = 0
    with r.Session() as rs:
        while end < t:
            end = time.time()-start
            if int(end/switch) != mid: # switch strategy
                ch = True; mid = int(end/switch)
            else:
                ch = False
            state, json = _strat(state, ch)
            l_thr = _clean(l_thr)
            if len(l_thr) > lim: # limit on concurrent requests
                if verbose:
                    print(f"Waiting... {len(l_thr)}",
                          end="          \r")
                time.sleep(random.randint(1, 3))
                continue
            l_thr.append(thr.Thread(target=requ,
                         args=(json, False, rs)))
            l_thr[-1].start()
            if verbose:
                print(f"{end:.03f}s: {len(l_thr)} requests,"+
                      f" state {state}.", 
                      end=" "*40+"\r")
            s = random.randint(0, 2001)/1000
            cur_rate = rate/max(len(l_thr), 1)
            time.sleep(s/cur_rate)
        if verbose:
            print(f"Exit main loop...") 
        for i, th in enumerate(l_thr):
            if verbose:
                print(f"Waiting for thread {i}...", end=" "*40+"\r")
            th.join()
    
def main(argv):
    t, ch_lim = 1200, 20.
    if len(sys.argv) > 1:
        t = int(sys.argv[1])
    if len(sys.argv) > 2:
        ch_lim = float(sys.argv[2])
    loop(t, ch_lim)

if __name__ == "__main__":
    main(sys.argv)