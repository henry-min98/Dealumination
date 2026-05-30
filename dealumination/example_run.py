"""
example_run.py
==============
Minimal example for running the dealumination kMC simulation
and plotting all three Al species concentrations over time.

Run from the dealumination/ directory:
    cd /Users/slee75/Desktop/dealumination/dealumination/dealumination
    python example_run.py

Available attributes after calc.calculate()
-------------------------------------------
Profiles (lists, recorded every spec_int.interval events):
    calc.t       -- simulation time [hours]
    calc.alf     -- NAl_f  (framework Al count) over time
    calc.alex    -- NAl_ex (extraframework monomer count) over time
    calc.alex2   -- dimer molecule count  = NAl_ex2 / 2  over time
    calc.ev      -- cumulative event count over time
    calc.hev     -- cumulative hydrolysis events over time
    calc.aev     -- cumulative aggregation events over time

Final scalars:
    calc.NAl          -- total Al atoms in system
    calc.NAl_f        -- final framework Al count
    calc.NAl_ex       -- final monomer count
    calc.NAl_ex2      -- final Al atoms in dimer state (dimer molecules = /2)
    calc.simutime     -- final simulation time [hours]
    calc.simevents    -- total kMC events executed
    calc.hydroevents  -- total hydrolysis events
    calc.aggreevents  -- total aggregation events
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Import the dealumination zmc package ────────────────────────────────────
from dealumination.zmc import Zmc          # adjust if your package path differs

# ── Simulation parameters ────────────────────────────────────────────────────
A_hydro = 0.023    # hydrolysis prefactor  [h^-1]
A_aggre = 52.103   # aggregation prefactor [h^-1]
D       = 20.0     # sigmoidal cutoff distance [Angstrom]
rho_Al  = 0.3499   # Al density [atoms / nm^3]

print("Initialising Zmc ...")
calc = Zmc(A_hydro=A_hydro,
           A_aggre=A_aggre,
           D=D,
           rho_Al=rho_Al,
           short=False)

print(f"  Total Al sites  : {calc.NAl}")
print(f"  Initial Al_f    : {calc.NAl_f}")
print(f"  Supercell volume: {calc.vol:.2f} Ang^3")
print(f"  Time unit       : hours  |  max_time = {calc.max_time} h")
print()

# ── Run the kMC simulation ───────────────────────────────────────────────────
print("Running kMC simulation ...")
calc.calculate()

print(f"  Simulation finished.")
print(f"  Total events    : {calc.simevents:,}")
print(f"  Hydrolysis evts : {calc.hydroevents:,}")
print(f"  Aggregation evts: {calc.aggreevents:,}")
print(f"  Final time      : {calc.simutime:.4f} h")
print(f"  Final NAl_f     : {calc.NAl_f}")
print(f"  Final NAl_ex    : {calc.NAl_ex}")
print(f"  Final Al_ex2 (dimers): {calc.NAl_ex2 // 2}")
print()

# ── Extract profiles ─────────────────────────────────────────────────────────
t      = np.array(calc.t)       # time [h]
alf    = np.array(calc.alf)     # NAl_f
alex   = np.array(calc.alex)    # NAl_ex
alex2  = np.array(calc.alex2)   # dimer count  (NAl_ex2 / 2)
N_tot  = calc.NAl               # for normalising to fraction if desired

# ── Plotting ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: raw counts -------------------------------------------------
ax = axes[0]
ax.plot(t, alf,   color='steelblue',  lw=2,   label=r'$\mathrm{Al_f}$')
ax.plot(t, alex,  color='darkorange', lw=2,   label=r'$\mathrm{Al_{ex}}$  (monomer)')
ax.plot(t, alex2, color='forestgreen',lw=2,   label=r'$\mathrm{Al_{ex,2}}$ (dimer)')
ax.set_xlabel('Time  [h]',    fontsize=13)
ax.set_ylabel('Count  [atoms or dimers]', fontsize=13)
ax.set_title('Species counts over time', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

# --- Right panel: fractions normalised by NAl --------------------------------
ax = axes[1]
ax.plot(t, alf  / N_tot, color='steelblue',   lw=2,  label=r'$\mathrm{Al_f}$ / $N_{Al}$')
ax.plot(t, alex / N_tot, color='darkorange',  lw=2,  label=r'$\mathrm{Al_{ex}}$ / $N_{Al}$')
ax.plot(t, alex2/ N_tot, color='forestgreen', lw=2,  label=r'$\mathrm{Al_{ex,2}}$ (dimers) / $N_{Al}$')
ax.set_xlabel('Time  [h]',  fontsize=13)
ax.set_ylabel('Fraction',   fontsize=13)
ax.set_title('Species fractions over time', fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('dealumination_species_profiles.png', dpi=150, bbox_inches='tight')
print("Plot saved to  dealumination_species_profiles.png")
plt.show()

# ── Optional: print a compact summary table ──────────────────────────────────
print("\n--- Species profile (every 5th recorded point) ---")
print(f"{'Time [h]':>12}  {'NAl_f':>8}  {'NAl_ex':>8}  {'Dimers':>8}")
print("-" * 46)
step = max(1, len(t) // 20)
for i in range(0, len(t), step):
    print(f"{t[i]:12.3f}  {alf[i]:8.0f}  {alex[i]:8.0f}  {alex2[i]:8.1f}")
