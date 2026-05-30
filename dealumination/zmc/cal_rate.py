import numpy as np
from scipy.interpolate import interp1d
from .Zmc import Zmc
from .monkeypatch import monkeypatch_class

@monkeypatch_class(Zmc)
def calc_alf_profile(self, t_query):
    """Interpolate simulated Al_f/u.c. at requested time points [hours].

    Parameters
    ----------
    t_query : array-like  -- time points [h]

    Returns
    -------
    numpy array of Al_f / u.c. values
    """
    t_sim   = np.array(self.t)
    alf_sim = np.array(self.alf) / self.N_uc   # convert to Al_f / u.c.

    if len(t_sim) < 2:
        return np.full(len(t_query), alf_sim[-1])

    f_interp = interp1d(t_sim, alf_sim, kind='linear',
                        bounds_error=False,
                        fill_value=(alf_sim[0], alf_sim[-1]))
    return f_interp(np.array(t_query))

@monkeypatch_class(Zmc)
def calc_frac(self):
    """Final Al_f fraction relative to initial NAl (dimensionless)."""
    return self.NAl_f / self.NAl if self.NAl > 0 else 0.0

@monkeypatch_class(Zmc)
def calc_dimer_frac(self):
    """Final dimer fraction: (NAl_ex2/2) / NAl (dimensionless)."""
    return (self.NAl_ex2 / 2) / self.NAl if self.NAl > 0 else 0.0
