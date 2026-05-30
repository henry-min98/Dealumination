import numpy as np
from .Zmc import Zmc
from .monkeypatch import monkeypatch_class

@monkeypatch_class(Zmc)
def calc_frac(self):
    raw_time = np.array(self.t)
    raw_CuII = np.array(self.tw)
    raw_tot  = self.NCu

    start = len(raw_time) // 2
    avg   = np.average(raw_CuII[start:] / raw_tot)

    return avg
