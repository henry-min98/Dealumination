import os
import math
import numpy as np
from random import seed
from random import choice
from timeit import default_timer as timer

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class interval:
    def __init__(self):
        self.label    = 'event'   # lowercase to match all comparisons in calculator.py
        self.interval = 10

class Aluminum:
    """Single Al site on the CHA lattice.
    state 0 = Al_f   (framework)
    state 1 = Al_ex  (extraframework monomer)
    state 2 = Al_ex2 (in dimer/aggregate; dimer molecule count = NAl_ex2 / 2)
    """
    def __init__(self, lab, state, pos):
        self.label    = lab
        self.state    = state
        self.position = pos   # numpy array, always a copy (see seed_cha_auto)


# ---------------------------------------------------------------------------
# Zmc class
# ---------------------------------------------------------------------------

class Zmc:
    """
    Dealumination kMC simulation on a CHA zeolite lattice.

    Events
    ------
    Hydrolysis  (i, 0) : Al_f(0)  -> Al_ex(1),  rate = A_hydro [h^-1]
    Aggregation (i, j) : Al_ex(1) + Al_ex(1) -> Al_ex2(2) + Al_ex2(2)
                         rate = A_aggre * sigmo(d) [h^-1]
                         NAl_ex2 counts individual Al atoms; dimers = NAl_ex2/2

    Lattice
    -------
    POSCAR_CHA = hexagonal conventional CHA unit cell (36 T-sites, 3 cages/u.c.)
    9x9x9 supercell -> N_uc = 729 unit cells, 26,244 T-sites total.
    rho_Al is in [Al / u.c.]; NAl = round(rho_Al * N_uc).
    All output concentrations are divided by N_uc -> [species / u.c.].

    Sampling
    --------
    spec_int is TIME-based (label='time'), matching SCR reference exactly.
    At simevents==100 the interval is adapted: interval = 10^b h (with a
    floor guard to prevent blowup from fast early events).
    Time stamps are recorded at n*interval (uniform grid), NOT simutime.

    Time unit: hours throughout.
    """

    def __init__(self, A_hydro, A_aggre, D, rho_Al, short=False):

        # --- Kinetic parameters -------------------------------------------
        self.A_hydro   = A_hydro
        self.A_aggre   = A_aggre
        self.dec_mode  = 'sigmoidal'
        self.Cut_sigmo = D
        self.Slo_sigmo = 5.0

        # --- CHA lattice: load POSCAR once, store atomsrep for reuse ------
        # Bug fix: previously the POSCAR was loaded 3 separate times
        # (in __init__, seed_cha_auto, and minimum_image_convention).
        # Now loaded once here; seed_cha_auto and minimum_image_convention
        # reuse self.atomsrep directly.
        self.al        = 9
        self.bl        = 9
        self.cl        = 9
        self.seed_mode = 'CHA'
        self.short     = short
        self.rseed     = 1510420280

        from ase.io import read as ase_read
        _atoms = ase_read(os.path.join(os.path.dirname(__file__), 'POSCAR_CHA'))
        del _atoms[[a.index for a in _atoms if a.symbol == 'O']]
        self.atomsrep = _atoms.repeat([self.al, self.bl, self.cl])
        self.vol      = self.atomsrep.get_volume()   # Angstrom^3

        # N_uc: number of hexagonal conventional unit cells in supercell
        # rho_Al [Al/u.c.] -> NAl = total Al atoms in supercell
        self.N_uc = self.al * self.bl * self.cl       # 729
        self.NAl  = round(rho_Al * self.N_uc)

        # Validate: NAl must be > 0
        if self.NAl <= 0:
            raise ValueError(
                f'rho_Al={rho_Al} gives NAl={self.NAl}. '
                f'Increase rho_Al (try 36/(Si_Al+1)).')


        # --- Simulation parameters (time in HOURS) ------------------------
        self.max_events = 50000000
        self.max_time   = 500.0    # hours; updated adaptively at event 100
        self.wtime      = 36000.0  # wall-clock seconds

        # --- Recording intervals ------------------------------------------
        self.snap_int          = interval()
        self.snap_int.label    = 'event'
        self.snap_int.interval = 1000

        self.proc_int          = interval()
        self.proc_int.label    = 'event'
        self.proc_int.interval = 1000000

        # TIME-based to produce uniform time grid matching experimental data.
        # Initial value is a placeholder; overridden adaptively at event 100.
        self.spec_int          = interval()
        self.spec_int.label    = 'time'
        self.spec_int.interval = 1.0   # hours placeholder

        # --- Species counters (all Al start as Al_f = state 0) ------------
        self.NAl_f   = self.NAl    # framework Al
        self.NAl_ex  = 0           # extraframework monomers
        self.NAl_ex2 = 0           # Al atoms in dimer state; dimers = /2

        # --- Event counters -----------------------------------------------
        self.simutime    = 0.0     # float: avoid int/float type mixing in time array
        self.simevents   = 0
        self.hydroevents = 0
        self.aggreevents = 0

        # --- Internal structures (populated in minimum_image_convention) --
        # Initialised here so AttributeError is never raised before calculate()
        self.Al_list      = []
        self.indices      = []
        self.p2ind        = {}
        self.pairs        = {}
        self.min_distance = {}
        self.frequency    = {}
        self.rate_dict    = {}
        self.rate_nums    = []
        self.statistics   = False

        # --- Clean up stale output files ----------------------------------
        for f in ['error_output.txt', 'raw_outputs.txt', 'event_outputs.txt']:
            if f in os.listdir():
                os.system('rm ' + f)

        self.seed_cha_auto()
        self.read_outputs()


    # -----------------------------------------------------------------------
    # Lattice seeding
    # -----------------------------------------------------------------------

    def seed_cha_auto(self):
        """Place NAl Al sites on the CHA lattice (all state=0, Al_f).
        Lowenstein's rule: no two Al closer than 4 Angstrom (MIC).

        Uses self.atomsrep loaded in __init__ -- no second POSCAR read.
        Positions stored as .copy() to avoid numpy view aliasing bugs.
        """
        nums = list(range(len(self.atomsrep)))
        seed(self.rseed)

        max_attempts = len(nums) * 100   # guard against infinite loop
        for i in range(1, self.NAl + 1):
            attempts = 0
            while True:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f'Could not place Al site {i}/{self.NAl} after '
                        f'{max_attempts} attempts. rho_Al may be too high '
                        f'for Lowenstein rule on this lattice.')
                ind  = choice(nums)
                flag = 0
                for at in self.indices:
                    d = self.atomsrep.get_distance(at, ind, mic=True)
                    if round(d, 2) < 4:
                        flag = 1
                        break
                if flag == 0:
                    break

            # Store position as an explicit copy to prevent numpy view aliasing
            self.Al_list.append(
                Aluminum(lab=i, state=0,
                         pos=self.atomsrep[ind].position.copy()))
            self.indices.append(ind)
            self.p2ind[tuple(self.atomsrep[ind].position)] = ind

    # -----------------------------------------------------------------------
    # Output file initializers (called from notebook if file output needed)
    # -----------------------------------------------------------------------

    def init_snap(self):
        with open('Box_state.txt', 'w') as f:
            f.write('##### 3D coordinates of Al sites #####\n')
            f.write('S.No.     X      Y     Z\n')
            for i, al in enumerate(self.Al_list, start=1):
                p = al.position
                f.write(f'{i}     {p[0]}      {p[1]}      {p[2]}\n')

    def init_spec(self):
        with open('Specie_numbers.txt', 'w') as f:
            f.write('####### Species concentrations (time in hours) #######\n')
            f.write('S.No  Time[h]  Events  Al_f  Al_ex  Al_ex2(dimers)\n')

    def init_proc(self):
        with open('Process_statistics.txt', 'w') as f:
            f.write('###### Process counters #######\n')
            f.write('S.No  Time[h]  Events  Hydrolysis  Aggregation\n')

    def read_outputs(self):
        # Filled by monkeypatch in calculator.py
        pass
