import numpy as np
from scipy.stats import linregress
from .Zmc import Zmc
from .monkeypatch import monkeypatch_class

@monkeypatch_class(Zmc)
def calc_rate(self):
    ev_time       = np.array(self.t)
    ev_red_events = np.array(self.rev)

    start = len(ev_time) // 2
    x     = ev_time[start:]
    y     = ev_red_events[start:]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r = slope / self.NCu

    return r
