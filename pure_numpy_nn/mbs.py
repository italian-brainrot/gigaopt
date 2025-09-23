import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal, ROUND_HALF_UP
from functools import partial
from typing import Any

import numpy as np
import torch


def format_number(number, n):
    """Rounds to n significant digits after the decimal point."""
    if number == 0: return 0
    if math.isnan(number) or math.isinf(number) or (not math.isfinite(number)): return number
    if n <= 0: raise ValueError("n must be positive")

    dec = Decimal(str(number))
    if dec.is_zero(): return 0
    if number > 10**n or dec % 1 == 0: return int(dec)

    if abs(dec) >= 1:
        places = n
    else:
        frac_str = format(abs(dec), 'f').split('.')[1]
        leading_zeros = len(frac_str) - len(frac_str.lstrip('0'))
        places = leading_zeros + n

    quantizer = Decimal('1e-' + str(places))
    rounded_dec = dec.quantize(quantizer, rounding=ROUND_HALF_UP)

    if rounded_dec % 1 == 0: return int(rounded_dec)
    return float(rounded_dec)

def _tofloatlist(x) -> list[float]:
    if isinstance(x, (int,float)): return [x]
    if isinstance(x, np.ndarray) and x.size == 1: return [float(x.item())]
    if isinstance(x, torch.Tensor) and x.numel() == 1: return [float(x.item())]
    return [float(i) for i in x]

class MBS:
    """Univariate optimization via grid search followed by multi-binary search, supports multi-objective functions, good for plotting.

    Args:
        grid (Iterable[float], optional): values for initial grid search.
        step (float, optional): expansion step size. Defaults to 1.
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
        rounding (int, optional): rounding is to significant digits, avoids evaluating points that are too close.
        log_scale (bool, optional):
            whether to minimize in log10 scale. If true, it is assumed that ``grid`` is given in log10 scale,
            and evaluated points are also stored in log10 scale.
    """

    def __init__(
        self,
        grid: Iterable[float],
        step: float,
        num_candidates: int = 3,
        num_binary: int = 20,
        num_expansions: int = 20,
        rounding: int| None = 2,
        log_scale: bool = False,
    ):
        self.objectives: dict[int, dict[float,float]] = {}
        """dictionary of objectives, each maps point (x) to value (v)"""

        self.evaluated: set[float] = set()
        """set of evaluated points (x)"""

        grid = tuple(grid)
        if len(grid) == 0: raise ValueError("At least one grid search point must be specified")
        self.grid = sorted(grid)

        self.step = step
        self.num_candidates = num_candidates
        self.num_binary = num_binary
        self.num_expansions = num_expansions
        self.rounding = rounding
        self.log_scale = log_scale

    def _get_best_x(self, n: int, objective: int):
        """n best points"""
        obj = self.objectives[objective]
        v_to_x = [(v,x) for x,v in obj.items()]
        v_to_x.sort(key = lambda vx: vx[0])
        xs = [x for v,x in v_to_x]
        return xs[:n]

    def _suggest_points_around(self, x: float, objective: int):
        """suggests points around x"""
        points = list(self.objectives[objective].keys())
        points.sort()
        if x not in points: raise RuntimeError(f"{x} not in {points}")

        expansions = []
        if x == points[0]:
            expansions.append((x-self.step, 'expansion'))

        if x == points[-1]:
            expansions.append((x+self.step, 'expansion'))

        if len(expansions) != 0: return expansions

        idx = points.index(x)
        xm = points[idx-1]
        xp = points[idx+1]

        x1 = (x - (x - xm)/2)
        x2 = (x + (xp - x)/2)

        return [(x1, 'binary'), (x2, 'binary')]

    def _evaluate(self, fn, x):
        """Evaluate a point, returns False if point is already in history"""
        if self.rounding is not None: x = format_number(x, self.rounding)
        if x in self.evaluated: return False
        self.evaluated.add(x)

        if self.log_scale: vals = _tofloatlist(fn(10 ** x))
        else: vals = _tofloatlist(fn(x))

        for idx, v in enumerate(vals):
            if idx not in self.objectives: self.objectives[idx] = {}
            self.objectives[idx][x] = v

        return True

    def run(self, fn):
        # step 1 - grid search
        for x in self.grid:
            self._evaluate(fn, x)

        # step 2 - binary search
        while True:
            if (self.num_candidates <= 0) or (self.num_expansions <= 0 and self.num_binary <= 0): break

            # suggest candidates
            candidates: list[tuple[float, str]] = []

            # sample around best points
            for objective in self.objectives:
                best_points = self._get_best_x(self.num_candidates, objective)
                for p in best_points:
                    candidates.extend(self._suggest_points_around(p, objective=objective))

            # filter
            if self.num_expansions <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'expansion']

            if self.num_candidates <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'binary']

            # if expansion was suggested, discard anything else
            types = [t for x, t in candidates]
            if any(t == 'expansion' for t in types):
                candidates = [(x,t) for x,t in candidates if t == 'expansion']

            # evaluate candidates
            terminate = False
            at_least_one_evaluated = False
            for x, t in candidates:
                evaluated = self._evaluate(fn, x)
                if not evaluated: continue
                at_least_one_evaluated = True

                if t == 'expansion': self.num_expansions -= 1
                elif t == 'binary': self.num_binary -= 1

                if self.num_binary < 0:
                    terminate = True
                    break

            if terminate: break
            if not at_least_one_evaluated:
                if self.rounding is None: break
                self.rounding += 1
                if self.rounding == 10: break

        # return dict[float, tuple[float,...]]
        ret = {}
        for i, objective in enumerate(self.objectives.values()):
            for x, v in objective.items():
                if self.log_scale: x = 10 ** x
                if x not in ret: ret[x] = [None for _ in self.objectives]
                ret[x][i] = v

        for v in ret.values():
            assert len(v) == len(self.objectives), v
            assert all(i is not None for i in v), v

        return ret

def mbs_minimize(fn, grid: Iterable[float], step:float, num_candidates: int = 3, num_binary: int = 20, num_expansions: int = 20, rounding=2, log_scale=False):
    mbs = MBS(grid, step=step, num_candidates=num_candidates, num_binary=num_binary, num_expansions=num_expansions, rounding=rounding, log_scale=log_scale)
    return mbs.run(fn)

