import os
import math
import numpy as np
import random as rng_mod
from timeit import default_timer as timer
from .Zmc import Zmc, Aluminum
from .monkeypatch import monkeypatch_class

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def find_ind(arr, r):
    """Select event index proportional to rates (Gillespie selection)."""
    Sumo = sum(arr)
    if Sumo == 0:
        return None
    Sumi = 0
    size = len(arr)
    if r > 0.5:
        Sum = Sumo
        for i in range(size - 1, -1, -1):
            if Sum <= r * Sumo:
                return i + 1
            Sum -= arr[i]
    else:
        for i in range(size):
            Sumi += arr[i]
            if Sumi >= r * Sumo:
                return i
    return 0

# ---------------------------------------------------------------------------
# Monkeypatched rate functions
# ---------------------------------------------------------------------------

@monkeypatch_class(Zmc)
def read_outputs(self):
    pass

@monkeypatch_class(Zmc)
def sigmo_function(self, d):
    """Sigmoidal distance-decay for aggregation."""
    m     = self.Slo_sigmo
    decay = 1.0 / (1.0 + np.exp(m * (d - 2.0 * self.Cut_sigmo)))
    return 0.0 if decay == 0 else decay

@monkeypatch_class(Zmc)
def calc_hydro_rate(self):
    """Hydrolysis: single-site, no distance dependence [h^-1]."""
    return self.A_hydro

@monkeypatch_class(Zmc)
def calc_aggre_rate(self, d):
    """Aggregation: distance-dependent [h^-1]."""
    r = self.A_aggre * self.sigmo_function(d)
    return r if r >= 1e-20 else 0.0


# ---------------------------------------------------------------------------
# MIC distance table + initial rate_dict
# Bug fix: uses self.atomsrep (loaded once in __init__) instead of
# re-reading POSCAR a third time.
# Bug fix A: aggregation adds ALL directed pairs (i,j) for every Al_ex,
# matching the SCR convention (not just undirected i<j pairs).
# ---------------------------------------------------------------------------

@monkeypatch_class(Zmc)
def minimum_image_convention(self):
    """Build pairwise MIC distance table and initial rate_dict.

    self.pairs[label] = {other_label: distance, ...}  for all other Al sites.
    self.rate_dict    ordered dict: key=(i,0) hydrolysis, key=(i,j) aggregation.

    Aggregation uses DIRECTED pairs: both (i,j) and (j,i) are added for
    every Al_ex pair, consistent with the SCR oxidation convention.
    """
    self.pairs        = {}
    self.min_distance = {}
    self.frequency    = {}
    self.rate_dict    = {}

    # Reuse self.atomsrep -- no POSCAR re-read here
    for c in self.Al_list:
        nn   = {}
        dees = []
        for u in self.Al_list:
            if u.label == c.label:
                continue
            d = self.atomsrep.get_distance(
                    self.p2ind[tuple(c.position)],
                    self.p2ind[tuple(u.position)],
                    mic=True)
            nn[u.label] = d
            dees.append(d)
        self.pairs[c.label]        = nn
        self.min_distance[c.label] = min(dees)
        self.frequency[c.label]    = []

    # Build initial rate_dict
    # t=0: all Al are Al_f -> only hydrolysis events exist
    for al in self.Al_list:
        if al.state == 0:
            self.rate_dict[tuple([al.label, 0])] = self.calc_hydro_rate()
        elif al.state == 1:
            # Directed pairs: add (al.label, nb_label) for ALL Al_ex neighbors
            for nb_label, d in self.pairs[al.label].items():
                if self.Al_list[nb_label - 1].state == 1:
                    r = self.calc_aggre_rate(d)
                    if r > 0:
                        self.rate_dict[tuple([al.label, nb_label])] = r
        # state 2: no events

    self.rate_nums = list(self.rate_dict.values())

@monkeypatch_class(Zmc)
def refresh_rlist(self):
    """Surgical update of rate_dict based on self.sites (last executed event).

    After hydrolysis (i, 0) -- site i: state 0 -> 1
      - remove (i, 0) hydrolysis entry
      - add directed aggregation pairs (i, j) and (j, i) for every
        current Al_ex neighbor j (state == 1)

    After aggregation (i, j) -- both sites: state 1 -> 2
      - remove every rate_dict entry that contains i or j
      - no new entries (state 2 has no events)
    """
    sites = self.sites

    if sites[1] == 0:
        # HYDROLYSIS: site i became Al_ex
        i = sites[0]
        # Remove its hydrolysis entry
        self.rate_dict.pop(tuple([i, 0]), None)
        # Add directed aggregation pairs with every current Al_ex neighbor
        for nb_label, d in self.pairs[i].items():
            if self.Al_list[nb_label - 1].state == 1:
                r = self.calc_aggre_rate(d)
                if r > 0:
                    self.rate_dict[tuple([i, nb_label])] = r
                    self.rate_dict[tuple([nb_label, i])] = r

    else:
        # AGGREGATION: both sites i, j became Al_ex2
        i, j = sites[0], sites[1]
        for x in list(self.rate_dict):
            if i in x or j in x:
                del self.rate_dict[x]
        # state 2 has no events -- nothing to add

    self.rate_nums = list(self.rate_dict.values())


@monkeypatch_class(Zmc)
def calculate(self):
    self.calc_dealumination()

@monkeypatch_class(Zmc)
def calc_dealumination(self):
    """
    Gillespie kMC loop for dealumination (transient, no steady state).

    Bug fixes applied here
    ----------------------
    A (directed pairs)  : refresh_rlist and minimum_image_convention now add
                          ALL directed (i,j) aggregation pairs, not just i<j.
    B (state guard)     : aggregation execution verifies both sites are still
                          state=1 before flipping; skips stale events safely.
    C (adaptive clamp)  : simutime at event 100 is clamped to >= 0.01 h
                          before log10 to prevent vanishingly small intervals.

    Sampling mirrors SCR calc_steady_state_standard_periodic exactly:
      - spec_int TIME-based; while-loop records at n*interval (uniform grid).
      - snap_int event-based (unchanged from SCR).
    """
    t_start = timer()
    rng_mod.seed(int(self.rseed))
    self.minimum_image_convention()

    # Output lists -- seed with t=0 initial state
    time         = [0.0]
    alf          = [self.NAl_f]
    alex         = [self.NAl_ex]
    alex2_dimers = [self.NAl_ex2 / 2]
    events       = [0]
    hydroevs     = [0]
    aggreevs     = [0]

    first = [al.state for al in self.Al_list]
    self.positions   = [first]
    self.snap_times  = [0.0]
    self.snap_events = [0]

    n_int      = 1
    n_int_snap = 1

    # -----------------------------------------------------------------------
    while True:

        # Wall-clock guard
        if (timer() - t_start) > self.wtime:
            break

        if len(self.rate_dict) == 0:
            break

        R_total = sum(self.rate_nums)
        if R_total < 1e-20:   # float-safe zero check: exact ==0 never triggers
            break

        # Select event
        r1  = rng_mod.uniform(0, 1)
        ind = find_ind(self.rate_nums, r1)
        if ind is None or ind >= len(self.rate_nums):
            break
        if self.rate_nums[ind] == 0:
            break

        # Advance clock [hours]: dt = -ln(r2) / R_total
        r2        = rng_mod.uniform(1e-15, 1)   # avoid log(0)
        exec_time = -1.0 * np.log(r2) / R_total
        self.simutime  += exec_time
        self.simevents += 1

        sites = list(self.rate_dict)[ind]
        self.sites = sites   # stored for surgical refresh_rlist

        # --- Execute event ------------------------------------------------
        if sites[1] == 0:
            # HYDROLYSIS: Al_f(i) -> Al_ex(i)
            i = sites[0]
            self.Al_list[i - 1].state = 1
            self.NAl_f  -= 1
            self.NAl_ex += 1
            self.hydroevents += 1
            self.frequency[i].append(self.simevents)

        else:
            # AGGREGATION: Al_ex(i) + Al_ex(j) -> Al_ex2(i) + Al_ex2(j)
            # Bug fix B: guard against stale rate_dict entries.
            # Both sites must still be state=1; if not, skip and refresh.
            i, j = sites[0], sites[1]
            if (self.Al_list[i - 1].state == 1 and
                    self.Al_list[j - 1].state == 1):
                self.Al_list[i - 1].state = 2
                self.Al_list[j - 1].state = 2
                self.NAl_ex  -= 2
                self.NAl_ex2 += 2        # dimer molecule count = NAl_ex2 / 2
                self.aggreevents += 1
                self.frequency[i].append(self.simevents)
                self.frequency[j].append(self.simevents)
            # else: stale event; refresh_rlist below will clean it up

        # --- Adaptive spec interval (mirrors SCR) -------------------------
        # Bug fix C: clamp simutime to >= 0.01 h before log10 so that
        # very fast early events (large A_aggre) cannot produce a
        # vanishingly small interval that terminates the run immediately.
        if self.spec_int.label == 'time':
            if self.simevents == 100:
                t_safe = max(self.simutime, 1e-2)   # floor at 0.01 h
                b = math.floor(math.log10(t_safe)) + 1
                time = [] ; alf = [] ; alex = [] ; alex2_dimers = []
                events = [] ; hydroevs = [] ; aggreevs = []
                n_int = 0
                self.spec_int.interval = 10 ** b
                self.max_time = 100 * self.spec_int.interval

        # --- Snap recording (event-based) ---------------------------------
        temp_pos = []
        if self.snap_int.label == 'event':
            if self.simevents >= n_int_snap * self.snap_int.interval:
                while self.simevents >= n_int_snap * self.snap_int.interval:
                    for al in self.Al_list:
                        temp_pos.append(al.state)
                    n_int_snap += 1
                self.snap_events.append(self.simevents)
                self.snap_times.append(self.simutime)
                self.positions.append(temp_pos)

        # --- Species recording (TIME-based while-loop) --------------------
        # Record at n*interval (uniform grid), not at self.simutime.
        # Cap: stop filling grid points once n*interval exceeds max_time
        # to prevent the while loop from generating millions of entries
        # when a single event jumps simutime by an astronomical amount.
        if self.spec_int.label == 'time':
            if self.simutime >= n_int * self.spec_int.interval:
                while self.simutime >= n_int * self.spec_int.interval:
                    if n_int * self.spec_int.interval > self.max_time:
                        break
                    time.append(n_int * self.spec_int.interval)
                    alf.append(self.NAl_f)
                    alex.append(self.NAl_ex)
                    alex2_dimers.append(self.NAl_ex2 / 2)
                    events.append(self.simevents)
                    hydroevs.append(self.hydroevents)
                    aggreevs.append(self.aggreevents)
                    n_int += 1

        # --- Rebuild rate list --------------------------------------------
        self.refresh_rlist()

        # --- Early-stop when no productive events remain ------------------
        # short=True: Bayesian optimisation fast path
        # always: once NAl_f=0 and <=1 Al_ex remain, aggregation is impossible
        if self.short and self.NAl_f == 0:
            break
        if self.NAl_f == 0 and self.NAl_ex <= 1:
            break

        # --- Hard caps ----------------------------------------------------
        if self.simevents >= self.max_events or self.simutime >= self.max_time:
            break

    # --- Store results ----------------------------------------------------
    self.t     = time
    self.alf   = alf
    self.alex  = alex
    self.alex2 = alex2_dimers
    self.ev    = events
    self.hev   = hydroevs
    self.aev   = aggreevs
