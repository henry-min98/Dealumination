import numpy as np
import statistics
from .Zmc import Zmc
from .monkeypatch import monkeypatch_class

@monkeypatch_class(Zmc)
def EWMA(self, t, x):
    """Exponentially-weighted moving average convergence check (adapted for NAl)."""
    nsize = self.NAl
    lam   = 0.10
    EWMA  = []
    for i in range(len(x)):
        if i == 0:
            temp = float(x[i])
        else:
            temp = lam * float(x[i-1]) + (1 - lam) * float(x[i])
        EWMA.append(temp)

    nums = [float(a) for a in x]
    sig  = np.std(nums)
    L    = 1
    UCL  = [EWMA[-1] + L*sig*np.sqrt(lam*(1-np.power((1-lam), i))/(2-lam))
            for i in range(len(x))]
    LCL  = [EWMA[-1] - L*sig*np.sqrt(lam*(1-np.power((1-lam), i))/(2-lam))
            for i in range(len(x))]

    first = None
    for i in range(len(x)):
        if EWMA[i] < UCL[i]:
            first = i
            break
    if first is None:
        return False

    pc = 0.10
    self.cutoff = first
    if first / len(x) >= pc:
        return False

    supsum = 0 ; flag = 0 ; covlist = [] ; conv = []
    for i in range(first, len(x)):
        supsum += nums[i] / nsize
        flag   += 1
        covlist.append(nums[i] / nsize)
        conv.append(nums[i])
    sd = statistics.stdev(conv)
    return (supsum / flag, flag, covlist, sd)

# ---------------------------------------------------------------------------
# Profile accessors -- all concentrations reported in [species / u.c.]
# ---------------------------------------------------------------------------

@monkeypatch_class(Zmc)
def get_alf_profile(self):
    """Return (time [h], Al_f / u.c.) arrays."""
    return np.array(self.t), np.array(self.alf) / self.N_uc

@monkeypatch_class(Zmc)
def get_dimer_profile(self):
    """Return (time [h], Al_ex2 dimers / u.c.) arrays.
    Dimer count per u.c. = (NAl_ex2 / 2) / N_uc.
    """
    return np.array(self.t), np.array(self.alex2) / self.N_uc

@monkeypatch_class(Zmc)
def get_species_profiles(self):
    """Return all three species profiles normalised to [species / u.c.].

    Keys
    ----
    'time'   : simulation time [h]
    'Al_f'   : Al_f / u.c.
    'Al_ex'  : Al_ex (monomer) / u.c.
    'Al_ex2' : Al_ex2 (dimer molecules) / u.c.  =  (NAl_ex2 / 2) / N_uc
    """
    return {
        'time'   : np.array(self.t),
        'Al_f'   : np.array(self.alf)   / self.N_uc,
        'Al_ex'  : np.array(self.alex)  / self.N_uc,
        'Al_ex2' : np.array(self.alex2) / self.N_uc,
    }
