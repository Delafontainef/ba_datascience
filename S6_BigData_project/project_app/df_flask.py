from flask import Flask, request, Response
from prometheus_client import CONTENT_TYPE_LATEST
from dotenv import load_dotenv
from df_metrics import Prom  # metrics
import df_code as code       # operations
# import multiprocessing as mp
import threading as thr
import psutil, os, time

app = Flask(__name__)
load_dotenv()           # still not sure what that does
PROC = psutil.Process() # "monitors" process
K_IO = thr.Lock()       # lock for I/O operations
# p_conn = None         # for multiprocessing Pipe
p = Prom(proc=PROC); p.set_reg()

@app.route("/", methods=['POST', 'GET'])
def call():
    kw = request.args if request.method == "GET" \
         else request.get_json(silent=True) or {}
    # p_conn.send(['before', kw])
    p._before(kw)
    kw = code.run(kw, K_IO)
    p._after(kw)
    # p_conn.send(['after', kw])
    return kw

@app.route("/metrics")
def metrics():
    # p_conn.send(["collect"])
    # res = p_conn.recv()
    res = {"result": p._collect()}
    if "result" in res:
        return Response(res['result'], mimetype=CONTENT_TYPE_LATEST)
    else:
        return Response(f"Error: {res['error']}\n", status=500,
                        mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # p_conn, c_conn = mp.Pipe()
    # p = Prom(); pp = mp.Process(target=p.run, args(c_conn, PROC))
    # pp.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
    # pp.terminate()
