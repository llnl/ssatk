"""Accepted-step constraints using SciPy's public ODE solver interface."""

import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, Radau, DenseOutput, OdeSolver


def projected_solver(method, project):
    """Wrap a SciPy solver with an accepted-state projection."""
    methods = {"RK23": RK23, "RK45": RK45, "DOP853": DOP853,
               "Radau": Radau, "BDF": BDF, "LSODA": LSODA}
    base = methods.get(method) if isinstance(method, str) else method
    if not isinstance(base, type) or not issubclass(base, OdeSolver):
        raise ValueError("method must be a SciPy method name or OdeSolver class.")

    class ProjectedDenseOutput(DenseOutput):
        def __init__(self, dense):
            super().__init__(dense.t_old, dense.t)
            self.dense = dense

        def _call_impl(self, times):
            values = self.dense(times)
            if times.ndim == 0:
                return project(float(times), values)
            if times.size == 0:
                return values
            return np.column_stack([project(t, y) for t, y in zip(times, values.T)])

    class ProjectedSolver(OdeSolver):
        def __init__(self, fun, t0, y0, t_bound, vectorized=False, **options):
            if vectorized:
                raise ValueError("projected wheel integration requires vectorized=False.")
            super().__init__(lambda t, y: fun(t, project(t, y)), t0, y0,
                             t_bound, vectorized=False)
            self.options = dict(options)
            self._start_solver(t0, y0, options)
            self.interpolant = None

        def _start_solver(self, t, y, options):
            self.solver = base(self.fun_single, t, y, self.t_bound,
                               vectorized=False, **options)
            for name in ("nfev", "njev", "nlu"):
                setattr(self, name, getattr(self, name) + getattr(self.solver, name))

        def _step_impl(self):
            solver = self.solver
            counters = {name: getattr(solver, name) for name in ("nfev", "njev", "nlu")}
            message = solver.step()
            if solver.status != "failed":
                self.interpolant = ProjectedDenseOutput(solver.dense_output())
            for name, before in counters.items():
                setattr(self, name, getattr(self, name) + getattr(solver, name) - before)
            if solver.status == "failed":
                return False, message
            self.t = solver.t
            self.y = project(self.t, solver.y)
            if not np.array_equal(self.y, solver.y) and self.t != self.t_bound:
                options = dict(self.options)
                options["first_step"] = min(solver.step_size, abs(self.t_bound - self.t))
                self._start_solver(self.t, self.y, options)
            return True, None

        def _dense_output_impl(self):
            return self.interpolant

    return ProjectedSolver
