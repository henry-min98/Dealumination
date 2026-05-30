import numpy as np
from .Zmc import Zmc
from .monkeypatch import monkeypatch_class

@monkeypatch_class(Zmc)
def calc_frac(self):
    """Return the final Al_f fraction relative to initial NAl.
    Useful as a quick scalar summary for convergence checks.
    """
    return self.NAl_f / self.NAl if self.NAl > 0 else 0.0

@monkeypatch_class(Zmc)
def calc_dimer_frac(self):
    """Return the final dimer (Al_ex2) fraction.
    NAl_ex2 / 2 gives dimer molecule count; divide by NAl for fraction.
    """
    return (self.NAl_ex2 / 2) / self.NAl if self.NAl > 0 else 0.0
