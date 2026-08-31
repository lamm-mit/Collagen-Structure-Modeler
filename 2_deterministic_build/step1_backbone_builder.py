"""
Faithful pure-Python port of THeBuScr (Rainey & Goh 2004) collagen builder.

This transpiles the ALGORITHMIC procedures of THeBuScr.1.07a.tcl one-for-one
(propensity lookup, triplet-Tm, boxcar smoothing, helix-type assignment, and the
cylindrical-polar geometry construction). Only the Tk GUI procedures are dropped.
No tclsh / tleap / AmberTools dependency.

It reproduces THeBuScr's batch mode exactly:
    atsel = "B"        -> backbone N, CA, C, O (no CB)
    coordtype = "A"    -> Cartesian
    resnum = "A"       -> residues numbered from 1
    prohyp = "B"       -> no automatic Pro(Y) -> Hyp relabelling
    helcutoff_numaa = 4
    smoothtrans = "N"  -> heltypes are integers 0/1 (single-residue transitions)
    helcutoff_tm = mean_tm

Verified bit-for-bit against the original THeBuScr on the 533-sequence set.

Tcl semantics preserved: `lindex` out of range -> 0.0 (`li`); Tcl "arrays" are
Python dicts keyed by int index; list `linsert ... 0` is insert(0, .), `lappend`
is append.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

PARAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thparams")
PROP_PATH = os.path.join(PARAM_DIR, "THpropensities.dat")

HELCUTOFF_NUMAA = 4


# ---------------------------------------------------------------------------
# Parameter / propensity loading (readparlist, readsmoothparlist, readpropensities)
# ---------------------------------------------------------------------------


def _read_parlist(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(";") or len(line) <= 3:
                continue
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) > 4:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def _read_smoothparlist(path: str) -> Dict[str, float]:
    """name index value -> {"name(index)": value}."""
    out: Dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(";") or len(line) <= 3:
                continue
            parts = line.split()
            if len(parts) >= 3 and len(parts[0]) > 4:
                try:
                    out[f"{parts[0]}({parts[1]})"] = float(parts[2])
                except ValueError:
                    pass
    return out


def _read_propensities(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(";") or len(line) <= 3:
                continue
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 3:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


PARAMS = {0: _read_parlist(os.path.join(PARAM_DIR, "THparams_AR.dat")),
          1: _read_parlist(os.path.join(PARAM_DIR, "THparams_IR.dat"))}
NONSMOOTH = _read_smoothparlist(os.path.join(PARAM_DIR, "THparams_nonsmooth.dat"))
PROPDATA = _read_propensities(PROP_PATH)


def _ns(kind: str, trip: str, chain: int) -> float:
    """nonsmooth_<kind>_<trip>(chain); missing -> 0.0 (Tcl would leave unset=0)."""
    return NONSMOOTH.get(f"nonsmooth_{kind}_{trip}({chain})", 0.0)


# ---------------------------------------------------------------------------
# Tcl helpers
# ---------------------------------------------------------------------------


def li(lst: List[float], i: int) -> float:
    """Tcl lindex with out-of-range -> 0.0."""
    return lst[i] if 0 <= i < len(lst) else 0.0


def ret_gxy(n: int) -> str:
    m = n % 3
    return "G" if m == 0 else ("X" if m == 1 else "Y")


# ---------------------------------------------------------------------------
# getprop / trippropensities
# ---------------------------------------------------------------------------


def getprop(trip: str) -> float:
    if trip[0] != "G":
        return 0.0
    if len(trip) == 3 and trip[2] == "P":          # string match ??P
        trip = trip[0:2] + "O"
    if trip in PROPDATA:
        return PROPDATA[trip]
    tripx = "G" + trip[1] + "O"
    tripy = "GP" + trip[2]
    if tripx in PROPDATA and tripy in PROPDATA and "GPO" in PROPDATA:
        return PROPDATA[tripx] + PROPDATA[tripy] - PROPDATA["GPO"]
    return 0.0


def trippropensities(cha: str, chb: str, chc: str):
    tripa = [getprop(cha[3 * i:3 * i + 3]) for i in range(len(cha) // 3)]
    tripb = [getprop(chb[3 * i:3 * i + 3]) for i in range(len(chb) // 3)]
    tripc = [getprop(chc[3 * i:3 * i + 3]) for i in range(len(chc) // 3)]
    # Accumulate sequentially (a, then b, then c) exactly as the Tcl does; the
    # floating-point rounding of this order is load-bearing, because setheltype
    # compares each Tm against mean_tm at the knife-edge of equality (uniform
    # sequences), where a 1e-14 difference flips the AR/IR assignment.
    mean_tm = 0.0
    for v in tripa:
        mean_tm += v
    for v in tripb:
        mean_tm += v
    for v in tripc:
        mean_tm += v
    mean_tm = mean_tm / (len(tripa) + len(tripb) + len(tripc))
    return tripa, tripb, tripc, mean_tm


# ---------------------------------------------------------------------------
# boxcarpropensities  (transcribed from THeBuScr.1.07a.tcl lines 402-1118)
# ---------------------------------------------------------------------------


def boxcarpropensities(cha: str, chb: str, chc: str, tripa, tripb, tripc):
    cha_tms: Dict[int, float] = {}
    chb_tms: Dict[int, float] = {}
    chc_tms: Dict[int, float] = {}
    cha_codes: Dict[int, str] = {}
    chb_codes: Dict[int, str] = {}
    chc_codes: Dict[int, str] = {}

    numta = len(cha) // 3
    numtb = len(chb) // 3
    numtc = len(chc) // 3
    if numtb < numtc:
        numtc = numtb
    if numtc < numtb:
        numtb = numtc

    cha_tms[0] = 0.0; cha_codes[0] = "G"
    cha_tms[1] = 0.0; cha_codes[1] = "X"
    chb_tms[0] = 0.0; chb_codes[0] = "G"

    a_num, b_num, c_num = 2, 1, 0
    Ymults = [24.0 / 9, 21.0 / 9, 12.0 / 9, 3.0 / 9]
    Xmults = [25.0 / 9, 18.0 / 9, 9.0 / 9, 1.0 / 9]
    Gmults = [24.0 / 9, 15.0 / 9, 6.0 / 9, 0.0]

    def setrow(tm):
        cha_tms[a_num] = tm; cha_codes[a_num] = ret_gxy(a_num)
        chb_tms[b_num] = tm; chb_codes[b_num] = ret_gxy(b_num)
        chc_tms[c_num] = tm; chc_codes[c_num] = ret_gxy(c_num)

    # --- N-terminal blocks (each: A/B/C get a diagonal triple of mult lists) ---
    def nterm_block(rng, amul, bmul, cmul):
        ct = 0.0; cc = 0.0
        for i in range(rng):
            ct += li(tripa, i) * li(amul, i)
            ct += li(tripb, i) * li(bmul, i)
            ct += li(tripc, i) * li(cmul, i)
            cc += li(Gmults, i); cc += li(Xmults, i); cc += li(Ymults, i)
        return ct / cc

    setrow(nterm_block(4, Ymults, Xmults, Gmults))                 # Y0A X0B G0C

    a_num += 1; b_num += 1; c_num += 1
    Gmults.insert(0, 21.0 / 9)
    setrow(nterm_block(4, Gmults, Ymults, Xmults))                 # G1A Y0B X0C

    a_num += 1; b_num += 1; c_num += 1
    Xmults.insert(0, 18.0 / 9); Ymults.append(0.0)
    setrow(nterm_block(5, Xmults, Gmults, Ymults))                 # X1A G1B Y0C

    a_num += 1; b_num += 1; c_num += 1
    Ymults.insert(0, 15.0 / 9)
    setrow(nterm_block(5, Ymults, Xmults, Gmults))                 # Y1A X1B G1C

    a_num += 1; b_num += 1; c_num += 1
    Gmults.insert(0, 12.0 / 9)
    setrow(nterm_block(5, Gmults, Ymults, Xmults))                 # G2A Y1B X1C

    a_num += 1; b_num += 1; c_num += 1
    Xmults.insert(0, 9.0 / 9)
    setrow(nterm_block(6, Xmults, Gmults, Ymults))                 # X2A G2B Y1C

    a_num += 1; b_num += 1; c_num += 1
    Ymults.insert(0, 6.0 / 9)
    setrow(nterm_block(6, Ymults, Xmults, Gmults))                 # Y2A X2B G2C

    a_num += 1; b_num += 1; c_num += 1
    Gmults.insert(0, 3.0 / 9)
    setrow(nterm_block(6, Gmults, Ymults, Xmults))                 # G3A Y2B X2C

    # --- middle: sliding window boxcar producing Tms0/1/2 ---
    endtrip = (numtb - 2) if numta > numtb else (numtb - 3)
    Tms0: List[float] = []; Tms1: List[float] = []; Tms2: List[float] = []
    for Trip in range(1, endtrip):
        startelm = Trip - 1
        endelm = Trip + 3
        Cha_vals = tripa[startelm:endelm + 1]
        Chb_vals = tripb[startelm:endelm + 1]
        Chc_vals = tripc[startelm:endelm + 1]
        Tmcurr = {}
        for WindPos in range(3):
            if WindPos == 0:
                Cha_coef = [1, 3, 3, 2, 0]; Chb_coef = [2, 3, 3, 1, 0]; Chc_coef = [3, 3, 3, 0, 0]
            elif WindPos == 1:
                Cha_coef = [0, 3, 3, 3, 0]; Chb_coef = [1, 3, 3, 2, 0]; Chc_coef = [2, 3, 3, 1, 0]
            else:
                Cha_coef = [0, 2, 3, 3, 1]; Chb_coef = [0, 3, 3, 3, 0]; Chc_coef = [1, 3, 3, 2, 0]
            Cha_cont = Chb_cont = Chc_cont = 0.0
            TotNumCont = 0.0
            for Pos in range(5):
                Cha_cont += li(Cha_coef, Pos) * li(Cha_vals, Pos)
                TotNumCont += li(Cha_coef, Pos) * (1 if li(Cha_vals, Pos) != 0 else 0)
                if Pos < 4:
                    Chb_cont += li(Chb_coef, Pos) * li(Chb_vals, Pos)
                    Chc_cont += li(Chc_coef, Pos) * li(Chc_vals, Pos)
                    TotNumCont += li(Chb_coef, Pos) * (1 if li(Chb_vals, Pos) != 0 else 0)
                    TotNumCont += li(Chc_coef, Pos) * (1 if li(Chc_vals, Pos) != 0 else 0)
            Tmcurr[WindPos] = (Cha_cont + Chb_cont + Chc_cont) / TotNumCont
        Tms0.append(Tmcurr[0]); Tms1.append(Tmcurr[1]); Tms2.append(Tmcurr[2])

    # add last two window positions manually if chain A == chain B
    if numtb == numta:
        startelm = numtb - 4
        Cha_vals = tripa[startelm:]
        Chb_vals = tripb[startelm:]
        Chc_vals = tripc[startelm:]
        for WindPos in range(2):
            if WindPos == 0:
                Cha_coef = [1, 3, 3, 2]; Chb_coef = [2, 3, 3, 1]; Chc_coef = [3, 3, 3, 0]
            else:
                Cha_coef = [0, 3, 3, 3]; Chb_coef = [1, 3, 3, 2]; Chc_coef = [2, 3, 3, 1]
            Cha_cont = Chb_cont = Chc_cont = 0.0
            TotNumCont = 0.0
            for Pos in range(4):
                Cha_cont += li(Cha_coef, Pos) * li(Cha_vals, Pos)
                Chb_cont += li(Chb_coef, Pos) * li(Chb_vals, Pos)
                Chc_cont += li(Chc_coef, Pos) * li(Chc_vals, Pos)
                TotNumCont += li(Cha_coef, Pos) * (1 if li(Cha_vals, Pos) != 0 else 0)
                TotNumCont += li(Chb_coef, Pos) * (1 if li(Chb_vals, Pos) != 0 else 0)
                TotNumCont += li(Chc_coef, Pos) * (1 if li(Chc_vals, Pos) != 0 else 0)
            v = (Cha_cont + Chb_cont + Chc_cont) / TotNumCont
            if WindPos == 0:
                Tms0.append(v)
            else:
                Tms1.append(v)

    # steady-state boxcar
    for Trip in range(3, numtb - 3):
        for WindPos in range(3):
            if WindPos == 0:
                CurrTm = (li(Tms0, Trip - 3) + li(Tms1, Trip - 3) + li(Tms2, Trip - 3)
                          + li(Tms0, Trip - 2) + li(Tms1, Trip - 2) + li(Tms2, Trip - 2)
                          + li(Tms0, Trip - 1) + li(Tms1, Trip - 1) + li(Tms2, Trip - 1))
            elif WindPos == 1:
                CurrTm = (li(Tms1, Trip - 3) + li(Tms2, Trip - 3)
                          + li(Tms0, Trip - 2) + li(Tms1, Trip - 2) + li(Tms2, Trip - 2)
                          + li(Tms0, Trip - 1) + li(Tms1, Trip - 1) + li(Tms2, Trip - 1)
                          + li(Tms0, Trip))
            else:
                CurrTm = (li(Tms2, Trip - 3)
                          + li(Tms0, Trip - 2) + li(Tms1, Trip - 2) + li(Tms2, Trip - 2)
                          + li(Tms0, Trip - 1) + li(Tms1, Trip - 1) + li(Tms2, Trip - 1)
                          + li(Tms0, Trip) + li(Tms1, Trip))
            CurrTm = CurrTm / 9
            a_num += 1; b_num += 1; c_num += 1
            setrow(CurrTm)

    # final boxcar if A longer than B
    if numta > numtb:
        Trip = numtb - 3
        CurrTm = (li(Tms0, Trip - 3) + li(Tms1, Trip - 3) + li(Tms2, Trip - 3)
                  + li(Tms0, Trip - 2) + li(Tms1, Trip - 2) + li(Tms2, Trip - 2)
                  + li(Tms0, Trip - 1) + li(Tms1, Trip - 1) + li(Tms2, Trip - 1)) / 9
        a_num += 1; b_num += 1; c_num += 1
        setrow(CurrTm)

    # --- C-terminal: work backwards ---
    if numta > numtb:
        cha_tms[(numta - 1) * 3 + 1] = 0.0; cha_codes[(numta - 1) * 3 + 1] = "X"
        cha_tms[(numta - 1) * 3 + 2] = 0.0; cha_codes[(numta - 1) * 3 + 2] = "Y"
        chc_tms[(numtc - 1) * 3 + 2] = 0.0; chc_codes[(numtc - 1) * 3 + 2] = "Y"
        a_num = (numta - 1) * 3
        b_num = (numtb - 1) * 3 + 2
        c_num = (numtc - 1) * 3 + 1
    elif numta == numtb:
        chb_tms[(numtb - 1) * 3 + 2] = 0.0; chb_codes[(numtb - 1) * 3 + 2] = "Y"
        chc_tms[(numtc - 1) * 3 + 1] = 0.0; chc_codes[(numtc - 1) * 3 + 1] = "X"
        chc_tms[(numtc - 1) * 3 + 2] = 0.0; chc_codes[(numtc - 1) * 3 + 2] = "Y"
        a_num = (numta - 1) * 3 + 2
        b_num = (numtb - 1) * 3 + 1
        c_num = (numtc - 1) * 3
    else:
        return (cha_tms, chb_tms, chc_tms, cha_codes, chb_codes, chc_codes)

    Gmults = [24.0 / 9, 21.0 / 9, 12.0 / 9, 3.0 / 9, 0.0]
    Ymults = [21.0 / 9, 24.0 / 9, 15.0 / 9, 6.0 / 9, 0.0]
    Xmults = [18.0 / 9, 25.0 / 9, 18.0 / 9, 9.0 / 9, 1.0 / 9]

    def cterm_block(rng, amul, bmul, cmul, longA):
        ct = 0.0; cc = 0.0
        if numta > numtb and longA:
            ct = li(tripa, numta - 1) * li(amul, 0)
            cc = li(amul, 0)
        for i in range(1, rng):
            ct += li(tripa, numtb - i) * li(amul, i)
            ct += li(tripb, numtb - i) * li(bmul, i)
            ct += li(tripc, numtb - i) * li(cmul, i)
            cc += li(Gmults, i); cc += li(Xmults, i); cc += li(Ymults, i)
        return ct / cc

    # A G(N+1), B Y(N), C X(N)  (only if numta>numtb)
    if numta > numtb:
        setrow(cterm_block(5, Gmults, Ymults, Xmults, longA=True))
        a_num -= 1; b_num -= 1; c_num -= 1

    # A Y(N), B X(N), C G(N)
    Gmults.insert(0, 15.0 / 9)
    setrow(cterm_block(5, Ymults, Xmults, Gmults, longA=True))

    a_num -= 1; b_num -= 1; c_num -= 1
    Ymults.insert(0, 12.0 / 9)
    setrow(cterm_block(5, Xmults, Gmults, Ymults, longA=True))     # A X(N) B G(N) C Y(N-1)

    a_num -= 1; b_num -= 1; c_num -= 1
    Xmults.insert(0, 9.0 / 9)
    setrow(cterm_block(6, Gmults, Ymults, Xmults, longA=True))     # A G(N) B Y(N-1) C X(N-1)

    a_num -= 1; b_num -= 1; c_num -= 1
    Gmults.insert(0, 6.0 / 9)
    setrow(cterm_block(6, Ymults, Xmults, Gmults, longA=True))     # A Y(N-1) B X(N-1) C G(N-1)

    a_num -= 1; b_num -= 1; c_num -= 1
    Ymults.insert(0, 3.0 / 9)
    setrow(cterm_block(6, Xmults, Gmults, Ymults, longA=True))     # A X(N-1) B G(N-1) C Y(N-2)

    a_num -= 1; b_num -= 1; c_num -= 1
    Xmults.insert(0, 1.0 / 9)
    setrow(cterm_block(7, Gmults, Ymults, Xmults, longA=True))     # A G(N-1) B Y(N-2) C X(N-2)

    a_num -= 1; b_num -= 1; c_num -= 1
    Gmults.insert(0, 0.0)
    setrow(cterm_block(7, Ymults, Xmults, Gmults, longA=True))     # A Y(N-2) B X(N-2) C G(N-2)

    # Finally A X(N-2) B G(N-2) C Y(N-3) if numta == numtb
    if numta == numtb:
        a_num -= 1; b_num -= 1; c_num -= 1
        Ymults.insert(0, 0.0)
        ct = 0.0; cc = 0.0
        for i in range(1, 7):
            ct += li(tripa, numtb - i) * li(Xmults, i)
            ct += li(tripb, numtb - i) * li(Gmults, i)
            ct += li(tripc, numtb - i) * li(Ymults, i)
            cc += li(Gmults, i); cc += li(Xmults, i); cc += li(Ymults, i)
        setrow(ct / cc)

    return (cha_tms, chb_tms, chc_tms, cha_codes, chb_codes, chc_codes)


# ---------------------------------------------------------------------------
# setheltype  (smoothtrans == "N": integer heltypes only)
# ---------------------------------------------------------------------------


def _assign_heltype(tms: Dict[int, float], mean_tm: float, gate_positive: bool) -> Dict[int, int]:
    n = len(tms)
    heltype: Dict[int, int] = {}
    starttype = 0
    currhelix = 0
    for i in range(n):
        currtm = tms[i]
        oldhelix = currhelix
        if gate_positive:
            currhelix = 0 if (currtm < mean_tm and currtm > 0) else 1
        else:
            currhelix = 0 if (currtm < mean_tm) else 1
        if currhelix != oldhelix:
            heltoset = oldhelix if (i - starttype) > HELCUTOFF_NUMAA else currhelix
            for j in range(starttype, i):
                heltype[j] = heltoset
            starttype = i
    heltoset = currhelix if (n - starttype) > HELCUTOFF_NUMAA else abs(currhelix - 1)
    for j in range(starttype, n):
        heltype[j] = heltoset
    return heltype


def setheltype(cha_tms, chb_tms, chc_tms, mean_tm):
    # chain A gates with `currtm > 0`; chains B and C do not (verbatim from tcl).
    return (_assign_heltype(cha_tms, mean_tm, True),
            _assign_heltype(chb_tms, mean_tm, False),
            _assign_heltype(chc_tms, mean_tm, False))


# ---------------------------------------------------------------------------
# makecylpolchain + outhelix  (atsel="B", coordtype="A", prohyp="B", resnum="A")
# ---------------------------------------------------------------------------


def _P(ht: int, name: str) -> float:
    return PARAMS[ht][name]


def _blend(m0: float, m1: float, ht0_name: str) -> float:
    return (m0 * _P(0, ht0_name) + m1 * _P(1, ht0_name)) / (m0 + m1)


def makecylpolchain(chainnum: int, heltype: Dict[int, int], currchain: str,
                    atsel: str = "B") -> List[Tuple[int, str, float, float, float]]:
    """Return list of (res_index, atom_name, x_axial, r, theta_deg)."""
    currmax = len(heltype)
    ht0 = heltype[0]
    x_off = (chainnum - 1) * _P(ht0, "CA_dX_Ch")
    cur_ang = (chainnum - 1) * _P(ht0, "CA_Ang_Ch")
    out: List[Tuple[int, str, float, float, float]] = []
    ht = heltype[0]

    def emit(res, atom, dx_name, r_name, ang_name, m0, m1):
        out.append((res, atom,
                    x_off + _blend(m0, m1, dx_name),
                    _blend(m0, m1, r_name),
                    cur_ang + _blend(m0, m1, ang_name)))

    for i in range(0, currmax, 3):
        # ---- Gly (i) ----
        oldht = ht; ht = heltype[i]
        m0 = abs(ht - 1); m1 = ht
        if ht != oldht:
            ov = oldht - ht
            x_off += ov * _ns("dX", "YG", chainnum)
            cur_ang += ov * _ns("Ang", "YG", chainnum)
        if i > 0:
            x_off += _blend(m0, m1, "CA_dX_Y_Gly")
            cur_ang += _blend(m0, m1, "CA_Ang_Y_Gly")
        if atsel != "C":
            emit(i, "N", "N_dX_Gly_toCA", "N_Gly_R", "N_Ang_Gly_toCA", m0, m1)
        out.append((i, "CA", x_off, _blend(m0, m1, "CA_Gly_R"), cur_ang))
        if atsel != "C":
            emit(i, "C", "CO_dX_Gly_toCA", "CO_Gly_R", "CO_Ang_Gly_toCA", m0, m1)
            emit(i, "O", "OC_dX_Gly_toCA", "OC_Gly_R", "OC_Ang_Gly_toCA", m0, m1)

        # ---- X (i+1) ----
        oldht = ht; ht = heltype[i + 1]
        m0 = abs(ht - 1); m1 = ht
        if ht != oldht:
            ov = oldht - ht
            x_off += ov * _ns("dX", "GX", chainnum)
            cur_ang += ov * _ns("Ang", "GX", chainnum)
        x_off += _blend(m0, m1, "CA_dX_Gly_X")
        cur_ang += _blend(m0, m1, "CA_Ang_Gly_X")
        if atsel != "C":
            emit(i + 1, "N", "N_dX_X_toCA", "N_X_R", "N_Ang_X_toCA", m0, m1)
        out.append((i + 1, "CA", x_off, _blend(m0, m1, "CA_X_R"), cur_ang))
        if atsel != "C":
            emit(i + 1, "C", "CO_dX_X_toCA", "CO_X_R", "CO_Ang_X_toCA", m0, m1)
            emit(i + 1, "O", "OC_dX_X_toCA", "OC_X_R", "OC_Ang_X_toCA", m0, m1)
        if atsel == "A" and currchain[i + 1] != "G":
            emit(i + 1, "CB", "CB_dX_X_toCA", "CB_X_R", "CB_Ang_X_toCA", m0, m1)

        # ---- Y (i+2) ----
        oldht = ht; ht = heltype[i + 2]
        m0 = abs(ht - 1); m1 = ht
        if ht != oldht:
            ov = oldht - ht
            x_off += ov * _ns("dX", "XY", chainnum)
            cur_ang += ov * _ns("Ang", "XY", chainnum)
        x_off += _blend(m0, m1, "CA_dX_X_Y")
        cur_ang += _blend(m0, m1, "CA_Ang_X_Y")
        if atsel != "C":
            emit(i + 2, "N", "N_dX_Y_toCA", "N_Y_R", "N_Ang_Y_toCA", m0, m1)
        out.append((i + 2, "CA", x_off, _blend(m0, m1, "CA_Y_R"), cur_ang))
        if atsel != "C":
            emit(i + 2, "C", "CO_dX_Y_toCA", "CO_Y_R", "CO_Ang_Y_toCA", m0, m1)
            emit(i + 2, "O", "OC_dX_Y_toCA", "OC_Y_R", "OC_Ang_Y_toCA", m0, m1)
        if atsel == "A" and currchain[i + 2] != "G":
            emit(i + 2, "CB", "CB_dX_Y_toCA", "CB_Y_R", "CB_Ang_Y_toCA", m0, m1)

    return out


# THeBuScr's outhelix uses a truncated pi literal (THeBuScr.1.07a.tcl line 1928),
# not full double precision. Using it keeps the trig bit-faithful to the oracle.
PI = 3.141592654


def _tcl_floor(x: float) -> float:
    """Tcl's floor(): returns a double, preserving the sign of zero
    (floor(-0.0) == -0.0). math.floor returns an int and loses that sign, which
    would flip a -0.0 axial angle to a spurious -0.000 coordinate."""
    return x if x == 0.0 else float(math.floor(x))


def _outhelix(chains_heltype, chains_seq, atsel="B") -> List[List[dict]]:
    """Assemble 3 chains, converting cylindrical-polar -> Cartesian exactly as
    THeBuScr's outhelix does (coordtype A): normalize theta to [0,360) then
    y = cos(pi*theta/180)*r, z = sin(pi*theta/180)*r, x = axial."""
    result = []
    for chainnum in (1, 2, 3):
        heltype = chains_heltype[chainnum - 1]
        seq = chains_seq[chainnum - 1]
        cyl = makecylpolchain(chainnum, heltype, seq, atsel=atsel)
        by_res: Dict[int, dict] = {}
        order: List[int] = []
        for res_idx, atom, x, r, theta in cyl:
            currz = theta - 360.0 * _tcl_floor(theta / 360.0)
            currz = currz + 360.0 * (1 if currz < 0 else 0)
            rad = PI * currz / 180.0
            y = math.cos(rad) * r
            z = math.sin(rad) * r
            if res_idx not in by_res:
                by_res[res_idx] = {"one_letter": seq[res_idx], "atoms": {}}
                order.append(res_idx)
            by_res[res_idx]["atoms"][atom] = (x, y, z)
        result.append([by_res[i] for i in order])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_triple_helix(sequences) -> List[List[dict]]:
    """sequences -> 3 chains (list of residue dicts). Homotrimer: 1 seq built x3;
    heterotrimer: 3 deposited chains. Reproduces THeBuScr batch mode exactly.

    Requires each chain already in Gly-X-Y register (multiple of 3, Gly first) and
    chains within one triplet of equal length — THeBuScr's own constraints. For
    arbitrary deposited sequences use build_collagen()."""
    seqs = [s.upper() for s in sequences]
    cha, chb, chc = (seqs * 3)[:3] if len(seqs) == 1 else seqs[:3]
    tripa, tripb, tripc, mean_tm = trippropensities(cha, chb, chc)
    cha_tms, chb_tms, chc_tms, *_ = boxcarpropensities(cha, chb, chc, tripa, tripb, tripc)
    ha, hb, hc = setheltype(cha_tms, chb_tms, chc_tms, mean_tm)
    return _outhelix((ha, hb, hc), (cha, chb, chc), atsel="B")


# ---------------------------------------------------------------------------
# Deposited-sequence layer (over the byte-exact THeBuScr core)
# ---------------------------------------------------------------------------
# THeBuScr requires each chain in Gly-X-Y register and only builds a gap-free Tm
# array when the chains are equal length (or A longer by one triplet). Deposited
# experimental chains start mid-triplet and differ by 1-2 residues. This layer
# reframes each chain to register, computes the AR/IR helix types on the common
# equal-length CORE with the byte-exact boxcar, then extends each longer chain's
# terminal helix type over its overhang so every residue is built (no truncation).
# The core is exactly THeBuScr; only the >1-triplet overhang is an extension.


def reframe(seq: str) -> str:
    """Trim a deposited chain to Gly-X-Y register: shift to the frame with the
    most Gly at position 0, then drop any trailing partial triplet."""
    s = seq.upper().strip().replace("﻿", "")
    f = min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))
    s = s[f:]
    return s[: len(s) - (len(s) % 3)]


def _extend_heltype(core: Dict[int, int], full_len: int) -> Dict[int, int]:
    h = dict(core)
    last = core[len(core) - 1] if core else 0
    for i in range(len(core), full_len):
        h[i] = last
    return h


def build_collagen(sequences, do_reframe: bool = True) -> List[List[dict]]:
    """Build a triple helix from deposited chain sequence(s), handling mid-triplet
    starts and unequal chain lengths (preserve-termini). Reduces to the byte-exact
    THeBuScr result when the (reframed) chains are equal length."""
    seqs = [s.upper() for s in sequences]
    cha, chb, chc = (seqs * 3)[:3] if len(seqs) == 1 else seqs[:3]
    if do_reframe:
        cha, chb, chc = reframe(cha), reframe(chb), reframe(chc)
    core = min(len(cha), len(chb), len(chc))
    core -= core % 3
    tripa, tripb, tripc, mean_tm = trippropensities(cha[:core], chb[:core], chc[:core])
    ta, tb_, tc, *_ = boxcarpropensities(cha[:core], chb[:core], chc[:core], tripa, tripb, tripc)
    ha_c, hb_c, hc_c = setheltype(ta, tb_, tc, mean_tm)
    ha = _extend_heltype(ha_c, len(cha))
    hb = _extend_heltype(hb_c, len(chb))
    hc = _extend_heltype(hc_c, len(chc))
    return _outhelix((ha, hb, hc), (cha, chb, chc), atsel="B")


# ---------------------------------------------------------------------------
# Pipeline entry point: build_backbone(sequences, mode)
# ---------------------------------------------------------------------------
# Two backbone variants used by the modular pipeline. Both reframe each chain to
# Gly-X-Y register; they differ only in how unequal reframed chain lengths are
# handled:
#   mode="core"   -> truncate all chains to the common equal-length core (the
#                    pure THeBuScr result; no unequal-length completion).
#   mode="extend" -> _extend_heltype completes chains that differ by whole
#                    triplets (identical to build_collagen above).
# Neither restores the residues reframe() trimmed off the termini — that is the
# job of step2_terminal_extension.extend_termini (the V3 "overhang" step).


def reframe_info(seq: str):
    """Return (frame_offset f, reframed_seq, n_dropped, c_dropped) for a chain.

    Mirrors reframe(): f residues are dropped at the N-terminus and the trailing
    partial triplet at the C-terminus. n_dropped / c_dropped are the actual
    one-letter residues removed (e.g. 'O', 'G'), needed to rebuild the termini."""
    s = seq.upper().strip().replace("﻿", "")
    f = min(range(3), key=lambda k: sum(1 for i in range(k, len(s), 3) if s[i] != "G"))
    shifted = s[f:]
    keep = len(shifted) - (len(shifted) % 3)
    reframed = shifted[:keep]
    n_dropped = s[:f]
    c_dropped = shifted[keep:]
    return f, reframed, n_dropped, c_dropped


def build_backbone(sequences, mode: str = "extend") -> List[List[dict]]:
    """Build the Phase-1 backbone (N/CA/C/O only) for a homo- or heterotrimer.

    mode="core"   : equal-length pure-port build (truncate to common core).
    mode="extend" : complete unequal (whole-triplet) chain lengths.
    Returns 3 chains as lists of residue dicts (thebuscr _outhelix format).
    """
    if mode not in ("core", "extend"):
        raise ValueError(f"mode must be 'core' or 'extend', got {mode!r}")

    seqs = [s.upper() for s in sequences]
    cha, chb, chc = (seqs * 3)[:3] if len(seqs) == 1 else seqs[:3]
    cha, chb, chc = reframe(cha), reframe(chb), reframe(chc)

    core = min(len(cha), len(chb), len(chc))
    core -= core % 3
    tripa, tripb, tripc, mean_tm = trippropensities(cha[:core], chb[:core], chc[:core])
    ta, tb_, tc, *_ = boxcarpropensities(cha[:core], chb[:core], chc[:core], tripa, tripb, tripc)
    ha_c, hb_c, hc_c = setheltype(ta, tb_, tc, mean_tm)

    if mode == "core":
        return _outhelix((ha_c, hb_c, hc_c),
                         (cha[:core], chb[:core], chc[:core]), atsel="B")

    # mode == "extend"
    ha = _extend_heltype(ha_c, len(cha))
    hb = _extend_heltype(hb_c, len(chb))
    hc = _extend_heltype(hc_c, len(chc))
    return _outhelix((ha, hb, hc), (cha, chb, chc), atsel="B")


# retthreelet: one-letter -> PDB comp_id (O -> HYP, as in THeBuScr).
_THREELET = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "O": "HYP", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR",
    "V": "VAL", "W": "TRP", "Y": "TYR",
}


def write_backbone_pdb(path: str, chains: List[List[dict]]) -> None:
    """Write the Phase-1 backbone PDB byte-for-byte in THeBuScr's `outtopdb`
    format (so Phase 2 / tleap behaves identically). Atom order N, CA, C, O;
    chains A/B/C; TER between chains; running serial counts TER lines."""
    lines = []
    atnum = 1
    for chlab, residues in zip("ABC", chains):
        resname = "UNK"
        resnum = 1
        for res in residues:
            resname = _THREELET.get(res["one_letter"], "UNK")
            for at in ("N", "CA", "C", "O", "CB"):
                if at not in res["atoms"]:
                    continue
                x, y, z = res["atoms"][at]
                element = at[0]
                lines.append(
                    f"ATOM  {atnum:5d}  {at:<3s}{resname:>4s}{chlab:>2s}{resnum:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}{element:>12s}  "
                )
                atnum += 1
            resnum += 1
        lines.append(f"TER   {atnum:5d}  {resname:>7s}{chlab:>2s}{resnum - 1:<58d}")
        atnum += 1
    lines.append(f"{'END':<80s}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Step 1: build the THeBuScr backbone (N/CA/C/O) from sequence(s).")
    ap.add_argument("--seq", action="append", required=True,
                    help="chain sequence (one-letter, O=HYP). Give once for a "
                         "homotrimer, or three times for a heterotrimer.")
    ap.add_argument("--mode", choices=("core", "extend"), default="extend")
    ap.add_argument("--out", required=True, help="output backbone PDB path")
    args = ap.parse_args()
    chains = build_backbone(args.seq, mode=args.mode)
    write_backbone_pdb(args.out, chains)
    print(f"wrote {args.out}  (mode={args.mode}, residues/chain="
          f"{[len(c) for c in chains]})")


if __name__ == "__main__":
    _main()
