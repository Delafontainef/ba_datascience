from prometheus_client import (
    generate_latest, CollectorRegistry,
    Counter, Gauge, Histogram
)
import random # DEBUG
import psutil, time

class Prom:
    """Tracks Flask application metrics.
       Built to be run on a subprocess for availability."""
    def __init__(self, proc=None, reg=None):
        self.op = {
            'before': self._before,
            'after': self._after,
            'collect': self._collect
        }
        self.proc = proc
        self.reg = reg
    
    def set_reg(self):
        """Sets up the CollectorRegistry."""
        self.reg = CollectorRegistry()
        self.mem = Gauge('proj_mem_bytes', "Memory usage.", registry=self.reg)
        self.cpu = Gauge('proj_cpu_percent', "CPU_usage.", registry=self.reg)
        self.thr = Gauge('proj_threads', "Number of threads.",
                         registry=self.reg)
        self.call_start = Counter('proj_call_start', 
                         "Number of calls started.", registry=self.reg)
        self.call_end = Counter('proj_call_end', "Number of calls ended.",
                         registry=self.reg)
        self.call_latency = Histogram('proj_call_latency',
                         "Latency by variable", ['size', 'cycle', 'io'],
                         buckets=[0., 1., 10., 100.],
                         registry=self.reg)
    def set_proc(self, pid):
        """Sets up process "monitoring"."""
        self.proc = psutil.Process(pid) if isinstance(pid, int) else pid
    def _set_labels(self, kw):
        size, cycle, io = kw['size'], kw['cycle'], kw['io']
        try:
           size = "low" if size < 100 else "mid" if size < 10000 \
                                      else "high"
        except Exception:
            size = "error"
        try:
           cycle = "low" if cycle < 100 else "mid" if cycle < 500 \
                                        else "high"
        except Exception:
            cycle = "error"
        # size = "low" if size < 100 else "mid" if size < 10000 \ 
        #                            else "high"
        # cycle = "low" if cycle < 100 else "mid" if cycle < 500 \
        #                              else "high"
        io = "True" if io else "False"
        # size = random.choice(['low', 'mid', 'high'])
        # cycle = random.choice(['low', 'mid', 'high'])
        # io = random.choice(['False', 'True'])
        return size, cycle, io
    def _before(self, kw):
        """Sets metrics before 'code.run()' is called."""
        self.call_start.inc()
    def _after(self, kw):
        """Sets metrics after 'code.run()' is called."""
        self.call_end.inc()
        size, cycle, io = self._set_labels(kw)
        self.call_latency.labels(size=size, cycle=cycle,
                                 io=io).observe(kw['time'])
    def _collect(self):
        """Generates metrics for scraper."""
        self.mem.set(self.proc.memory_info().rss)
        self.cpu.set(self.proc.cpu_percent(interval=0.1))
        self.thr.set(self.proc.num_threads())
        return generate_latest(self.reg).decode('utf-8')
    def run(self, conn, pid):
        """Endless loop to wait on 'conn' (Pipe) messages."""
        if self.reg == None:
            self.set_reg()
        if self.proc == None:
            self.set_proc(pid)
        while True:
            try:
                msg = conn.recv()
                res = self.op[msg[0]](*msg[1:])
                if msg[0] == "collect":
                    conn.send({"result": res})
            except Exception as e:
                conn.send({"error": str(e)}); break
    
