"""Two-point orbital transfer planning and propagation for SSAPy.

Given two boundary states (departure and arrival), this module solves
Lambert's problem for the connecting transfer orbit, expresses the two
required burns as delta-v vectors in the NTW frame (SSAPy convention:
N = in-plane normal to velocity, T = along velocity, W = along orbital
angular momentum), and optionally propagates the maneuver using SSAPy's
``AccelConstNTW`` finite-burn accelerations on top of any force model.

The force model defaults to two-body Keplerian gravity (``AccelKepler``)
but any SSAPy ``Accel`` (or list of them, e.g. harmonics + drag + SRP)
may be supplied, in which case the burns are computed against the
Keplerian Lambert solution and the trajectory is propagated under the
full force model.

Example
-------
>>> import numpy as np
>>> from ssapy.orbit import Orbit
>>> from ssapy.constants import RGEO, EARTH_MU
>>> r1 = np.array([7000e3, 0, 0])
>>> v1 = np.array([0, np.sqrt(EARTH_MU / 7000e3), 0])
>>> dep = Orbit(r1, v1, t=0.0)
>>> # ... build an arrival Orbit `arr` at a later time ...
>>> result = transfer_ssapy(dep, arr)
>>> print(result["delta_v_total"])
"""

import numpy as np

from ssapy.accel import AccelKepler, AccelConstNTW
from ssapy.orbit import Orbit
from ssapy.propagator import RK78Propagator
from ssapy.utils import normed, rv_to_ntw
from ssapy.constants import EARTH_MU
from ssatk.constants import G0
from ssatk.orbital_mechanics._transfer_result import _is_state_vector, maneuver_burn, trajectory_dict, transfer_boundary_states, transfer_result


_ARRIVAL_MODE_ALIASES = {
    "inject": "inject",
    "injection": "inject",
    "intercept": "intercept",
    "flyby": "intercept",
    "position": "intercept",
    "position_only": "intercept",
    "no_arrival_burn": "intercept",
    "rendezvous": "rendezvous",
    "match": "rendezvous",
    "match_state": "rendezvous",
    "match_velocity": "rendezvous",
    "insert": "insertion",
    "insertion": "insertion",
    "orbit_insert": "insertion",
    "orbit_insertion": "insertion",
    "target_orbit": "insertion",
}


def _is_state_tuple_without_time(value):
    if isinstance(value, Orbit) or np.isscalar(value) or isinstance(value, dict):
        return False
    try:
        values = tuple(value)
    except TypeError:
        return False
    return len(values) == 2 and all(_is_state_vector(item) for item in values)


def _normalize_keyword(value, aliases, name):
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{name} must be one of: {valid}") from exc


def _arrival_burn_for_mode(arrival_mode):
    if arrival_mode in ("rendezvous", "insertion"):
        return True
    if arrival_mode in ("inject", "intercept"):
        return False
    raise AssertionError(f"Unhandled arrival_mode {arrival_mode!r}")


def _arrival_mode_from_fixed_time_flags(arrival_burn):
    return "rendezvous" if arrival_burn else "intercept"


# ---------------------------------------------------------------------------
# Lambert solver (universal variables, Bate–Mueller–White / Vallado)
# ---------------------------------------------------------------------------

def _stumpff_C(z):
    if z > 1e-7:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < -1e-7:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 1 / 2 - z / 24 + z**2 / 720


def _stumpff_S(z):
    if z > 1e-7:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    if z < -1e-7:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    return 1 / 6 - z / 120 + z**2 / 5040


def solve_lambert(r1, r2, tof, mu=EARTH_MU, prograde=True, max_iter=200,
                  tol=1e-9):
    """Solve Lambert's problem with universal variables.

    Find the velocities of the (zero-revolution) conic connecting position
    ``r1`` to position ``r2`` in time-of-flight ``tof``.

    Parameters
    ----------
    r1, r2 : array_like, shape (3,)
        Initial and final position vectors [m] in an inertial frame.
    tof : float
        Time of flight [s]; must be > 0.
    mu : float
        Gravitational parameter [m^3/s^2].  Default is Earth's.
    prograde : bool
        If True, choose the short-way/prograde geometry (transfer angular
        momentum has +z component); otherwise retrograde.
    max_iter : int
        Maximum bisection/Newton iterations on the universal variable.
    tol : float
        Relative tolerance on time of flight.

    Returns
    -------
    v1, v2 : ndarray, shape (3,)
        Required velocity [m/s] at ``r1`` (departure) and ``r2`` (arrival)
        on the transfer conic.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge (e.g. near-180-degree transfer
        geometry, which is singular for single-plane Lambert solutions).
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if tof <= 0:
        raise ValueError("Time of flight must be positive.")

    r1n, r2n = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0)
    cross12 = np.cross(r1, r2)

    # Transfer angle, choosing branch from the requested motion sense.
    dnu = np.arccos(cos_dnu)
    if (prograde and cross12[2] < 0) or (not prograde and cross12[2] >= 0):
        dnu = 2 * np.pi - dnu

    if abs(np.sin(dnu)) < 1e-6 and np.cos(dnu) < 0:
        raise RuntimeError(
            "Lambert geometry is singular (transfer angle ~ 180 deg); "
            "the transfer plane is undefined. Adjust the boundary times "
            "or states slightly."
        )
    A = np.sin(dnu) * np.sqrt(r1n * r2n / (1 - np.cos(dnu)))

    def y(z):
        return r1n + r2n + A * (z * _stumpff_S(z) - 1) / np.sqrt(_stumpff_C(z))

    def tof_of_z(z):
        yy = y(z)
        if yy < 0:
            return None
        chi = np.sqrt(yy / _stumpff_C(z))
        return (chi**3 * _stumpff_S(z) + A * np.sqrt(yy)) / np.sqrt(mu)

    # Bracket z: lower bound where y(z) > 0, upper bound where tof exceeded.
    z_lo = -4 * np.pi**2
    while y(z_lo) < 0:
        z_lo /= 2
        if z_lo > -1e-12:
            break
    z_hi = 4 * np.pi**2 * 0.999  # below the z of an exactly closed orbit
    t_hi = tof_of_z(z_hi)
    if t_hi is not None and t_hi < tof:
        raise RuntimeError(
            "Requested time of flight exceeds the zero-revolution maximum "
            "for this geometry; a multi-revolution transfer would be "
            "required, which this solver does not implement."
        )

    # Bisection on z (robust; tof is monotonic in z for 0-rev case).
    for _ in range(max_iter):
        z = 0.5 * (z_lo + z_hi)
        t = tof_of_z(z)
        if t is None or t < tof:
            z_lo = z
        else:
            z_hi = z
        if t is not None and abs(t - tof) < tol * tof:
            break
    else:
        raise RuntimeError("Lambert solver failed to converge.")

    yy = y(z)
    f = 1 - yy / r1n
    g = A * np.sqrt(yy / mu)
    gdot = 1 - yy / r2n
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class Burn:
    """A single finite (or effectively impulsive) maneuver.

    Attributes
    ----------
    t_start, t_end : float
        Burn on/off times, GPS seconds.
    dv : ndarray (3,)
        Delta-v vector in the inertial frame [m/s].
    dv_ntw : ndarray (3,)
        Delta-v resolved on the NTW frame of the pre-burn state [m/s],
        ordered (N, T, W) per SSAPy convention.
    dv_mag : float
        Delta-v magnitude [m/s].
    direction_ntw : ndarray (3,)
        Unit thrust direction in NTW.
    accel : ssapy.accel.AccelConstNTW
        Ready-to-use SSAPy acceleration implementing this burn.
    """

    def __init__(self, t_start, t_end, dv, dv_ntw):
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.dv = np.asarray(dv, dtype=float)
        self.dv_ntw = np.asarray(dv_ntw, dtype=float)
        self.dv_mag = float(np.linalg.norm(dv))
        self.direction_ntw = (self.dv_ntw / self.dv_mag
                              if self.dv_mag > 0 else np.zeros(3))
        duration = self.t_end - self.t_start
        self.accel = AccelConstNTW(
            self.dv_ntw / duration,
            time_breakpoints=[self.t_start, self.t_end],
        )

    def __repr__(self):
        n, t, w = self.dv_ntw
        return (f"Burn(t={self.t_start:.1f} GPS s, |dv|={self.dv_mag:.3f} "
                f"m/s, NTW=[{n:.3f}, {t:.3f}, {w:.3f}] m/s)")


class TransferResult:
    """Output of :func:`transfer_ssapy`.

    Attributes
    ----------
    burns : list of Burn
        Departure and arrival burns.
    dv_total : float
        Sum of burn magnitudes [m/s].
    dv_budget : float or None
        The budget supplied by the caller, if any.
    within_budget : bool or None
        Whether ``dv_total`` <= ``dv_budget`` (None if no budget given).
    transfer_orbit : ssapy.orbit.Orbit
        Keplerian transfer conic immediately after the departure burn.
    trajectory : dict or None
        If propagated: ``{'t', 'r', 'v'}`` arrays sampled along the
        transfer under the full force model, including the finite burns.
    arrival_error : float or None
        If propagated: distance [m] between the propagated final position
        and the requested arrival position.
    propagator : ssapy.propagator.RK78Propagator or None
        The propagator (force model + burns) used, reusable for further
        calls to ``ssapy.compute.rv``.
    """

    def __init__(self, burns, dv_budget, transfer_orbit):
        self.burns = burns
        self.dv_total = float(sum(b.dv_mag for b in burns))
        self.dv_budget = dv_budget
        self.within_budget = (None if dv_budget is None
                              else bool(self.dv_total <= dv_budget))
        self.transfer_orbit = transfer_orbit
        self.trajectory = None
        self.arrival_error = None
        self.propagator = None

    def summary(self):
        lines = []
        for i, b in enumerate(self.burns, start=1):
            n, t, w = b.dv_ntw
            line = (f"Burn {i}: t = {b.t_start:.1f} GPS s, "
                    f"|dv| = {b.dv_mag:.3f} m/s "
                    f"({b.t_end - b.t_start:.1f} s), "
                    f"NTW dv = [N {n:+.3f}, T {t:+.3f}, W {w:+.3f}] m/s")
            if getattr(b, "propellant_mass", None) is not None:
                line += f", propellant ~{b.propellant_mass:.2f} kg"
            lines.append(line)
        lines.append(f"Total delta-v: {self.dv_total:.3f} m/s")
        if self.dv_budget is not None:
            status = "WITHIN" if self.within_budget else "EXCEEDS"
            lines.append(
                f"Budget: {self.dv_budget:.3f} m/s -> {status} budget")
        if self.arrival_error is not None:
            lines.append(
                f"Propagated arrival position error: "
                f"{self.arrival_error:.1f} m")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transfer_ssapy(
    *args,
    orbit1=None,
    orbit2=None,
    departure=None,
    arrival=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    t1=0.0,
    t2=None,
    tof=None,
    accel=None,
    dv_budget=None,
    burn_duration=10.0,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    prograde=True,
    arrival_mode=None,
    arrival_burn=True,
    propagate=True,
    propagator=None,
    refine=True,
    refine_tol=(10.0, 0.01),
    max_refine_iter=8,
    n_samples=200,
    rk_step=10.0,
    raise_on_budget=False,
):
    """Plan and propagate a two-burn transfer between two orbital states.

    Solves Lambert's problem between the departure position at its epoch
    and the arrival position at its (later) epoch, computes the departure
    and arrival delta-v vectors and their NTW components, models each burn
    as a constant-acceleration ``AccelConstNTW`` of length
    ``burn_duration``, and (optionally) propagates the maneuvering
    trajectory under the supplied force model with SSAPy's RK78 integrator.

    Parameters
    ----------
    departure, initial, orbit1 : ssapy.orbit.Orbit or tuple
        Pre-burn state: an SSAPy ``Orbit``, or a ``(r, v, t)`` tuple with
        ``r`` and ``v`` in meters / meters-per-second (inertial frame) and
        ``t`` in GPS seconds or as an ``astropy.time.Time``.
    arrival, target, orbit2 : ssapy.orbit.Orbit or tuple
        Target state in the same formats.  Its epoch must be later than
        the departure epoch; the difference sets the time of flight.  The
        target velocity defines the second (circularization/matching)
        burn.
    r1, v1, r2, v2 : array_like, optional
        Raw state-vector form for the same boundary conditions.  Equivalent
        positional form ``transfer_ssapy(r1, v1, r2, v2, tof=...)`` is also
        accepted.  Raw vector calls require ``tof`` or ``t2`` because Lambert
        transfers are fixed-time boundary-value problems.
    accel : ssapy.accel.Accel or list of Accel, optional
        Force model for propagation.  Default ``AccelKepler()`` (two-body
        gravity).  A list is summed.  Note the burn solution itself is the
        Keplerian Lambert solution; with strong perturbations the
        propagated arrival error reported in the result quantifies the
        mismatch.
    dv_budget : float, optional
        Total delta-v budget [m/s].  The result records whether the
        transfer fits; set ``raise_on_budget=True`` to raise instead.
    burn_duration : float
        Length of each finite burn [s].  Short durations approximate
        impulsive burns.  Ignored when ``thrust``/``mass`` are given.
        The burns must fit inside the time of flight.
    burn_accel : float, optional
        Burn acceleration magnitude [m/s^2] -- the simple hardware
        shortcut when the thrust-to-mass analysis was already done
        elsewhere.  Equivalent to specifying ``thrust``/``mass`` with
        the same ratio (and mutually exclusive with them): burn
        durations are sized as ``|dv_i| / burn_accel`` with the same
        consistency iteration and fit checks.  Propellant estimates are
        unavailable in this mode (those need ``mass`` and ``isp``).
    thrust : float, optional
        Engine thrust [N].  Requires ``mass``.  When given, each burn's
        duration is sized from the hardware instead of ``burn_duration``:
        ``duration_i = mass * |dv_i| / thrust`` (constant acceleration
        ``thrust/mass``; durations are fixed from the impulsive Lambert
        delta-vs, which the refinement then perturbs by < ~1%).  Raises
        ``ValueError`` when the engine is too weak for the burns to fit
        inside the time of flight.
    mass : float, optional
        Spacecraft wet mass [kg], treated as constant: propellant
        depletion is NOT modeled, so the burn acceleration does not grow
        as propellant is spent (a few-percent-of-mass burn is well
        approximated; a 30%+ burn is not).
    isp : float, optional
        Specific impulse [s].  Requires ``mass``.  Reporting only -- the
        dynamics are unchanged: attaches a Tsiolkovsky propellant-mass
        estimate to each burn (sequentially debited between burns) and
        to the summary.
    prograde : bool
        Sense of motion for the Lambert geometry.
    arrival_mode : {"inject", "intercept", "rendezvous", "insertion"}, optional
        User-facing terminology for the arrival constraint. ``"inject"`` means
        perform only the departure burn that places the spacecraft on the
        transfer conic. ``"intercept"`` means reach the target position at the
        specified time without matching velocity. ``"rendezvous"`` means reach
        the target position at the specified time and match velocity.
        ``"insertion"`` means include the arrival/insertion burn needed to enter
        the final orbit. This fixed-time solver still requires a target epoch;
        use ``transfer_optimal(arrival_mode="insertion")`` when target phase is
        a free search variable.
    arrival_burn : bool
        Lower-level compatibility flag used when ``arrival_mode`` is not set. If
        True (default), perform and cost the second burn that matches the arrival
        velocity. If False, return a single-burn fixed-time intercept/injection
        trajectory with expected arrival velocity mismatch.
    propagate : bool
        If True, numerically propagate the burn-inclusive trajectory and
        report the achieved arrival error.
    propagator : class or callable, optional
        How to build the propagator for each transfer segment.  Default
        is ``RK78Propagator`` with step ``rk_step``.  Accepts either a
        propagator *class* with an ``(accel, h)`` constructor (e.g.
        ``RK4Propagator``, ``RK8Propagator``, ``LeapfrogPropagator``) or
        a *factory* callable ``f(accel) -> propagator instance`` for full
        control, e.g.
        ``lambda a: SciPyPropagator(a, ode_kwargs=dict(rtol=1e-10))``.
        A class/factory is required rather than an instance because each
        segment (burn 1 / coast / burn 2) is propagated under a different
        acceleration sum.  The propagator must actually integrate the
        supplied accelerations: analytic/element propagators such as
        ``KeplerianPropagator``, ``SGP4Propagator``, or
        ``SeriesPropagator`` ignore them, so the burns would never fire
        (this is detected and raised during refinement).
    refine : bool
        If True (default, requires ``propagate``), apply a differential-
        correction (Newton shooting) step: the six NTW burn components
        are adjusted until the *propagated* finite-burn trajectory under
        the full force model meets the target position and velocity.
        This removes both the finite-burn/impulsive mismatch (the NTW
        frame rotates during a burn) and, when perturbing accelerations
        are supplied, the Keplerian-Lambert modeling error.
    refine_tol : tuple of float
        Convergence tolerances ``(position [m], velocity [m/s])`` for the
        refinement.
    max_refine_iter : int
        Maximum Newton iterations for the refinement.
    n_samples : int
        Number of trajectory sample times if propagating.
    rk_step : float
        Initial RK78 step size [s].
    raise_on_budget : bool
        If True and the budget is exceeded, raise ``ValueError``.
    Returns
    -------
    dict
        Canonical transfer dictionary with ``initial``, ``target``,
        ``burns``, ``trajectory``, ``delta_v_total``, hardware metadata,
        assumptions, and diagnostics.

    Notes
    -----
    * NTW components follow SSAPy's convention (``ssapy.utils.rv_to_ntw``):
      T along velocity, W along the orbital angular momentum (r x v), and
      N = T x W completing the right-handed, in-plane axis (radially
      outward for a circular orbit).
    * Burn 1 fires over ``[t1, t1 + d1]`` and burn 2 over
      ``[t2 - d2, t2]`` where the ``d_i`` are ``burn_duration`` or the
      thrust-sized durations; the differential correction absorbs the
      finite-burn arc so the propagated trajectory still meets the
      target to ``refine_tol``.
    """
    # --- normalize inputs ------------------------------------------------
    raw_four_vector_state = len(args) == 4 and all(_is_state_vector(value) for value in args)
    raw_tuple_state_pair = (
        len(args) == 2
        and all(_is_state_tuple_without_time(value) for value in args)
    )
    if (raw_four_vector_state or raw_tuple_state_pair) and t2 is None and tof is None:
        raise ValueError("transfer_ssapy raw-vector input requires tof/t2")

    if accel is None:
        accel = AccelKepler()
    elif isinstance(accel, (list, tuple)):
        accel = sum(accel[1:], accel[0])

    # Gravitational parameter for state inference and the Lambert conic.
    mu = getattr(accel, "mu", EARTH_MU)

    if arrival_mode is None:
        arrival_mode = _arrival_mode_from_fixed_time_flags(arrival_burn)
    else:
        arrival_mode = _normalize_keyword(arrival_mode, _ARRIVAL_MODE_ALIASES, "arrival_mode")
        arrival_burn = _arrival_burn_for_mode(arrival_mode)

    departure_state, arrival_state = transfer_boundary_states(
        *args,
        orbit1=orbit1,
        orbit2=orbit2,
        departure=departure,
        arrival=arrival,
        initial=initial,
        target=target,
        r1=r1,
        v1=v1,
        r2=r2,
        v2=v2,
        t1=t1,
        t2=t2,
        tof=tof,
        mu=mu,
        name="transfer_ssapy",
    )
    r1, v1, t1 = departure_state["r"], departure_state["v"], departure_state["t"]
    r2, v2, t2 = arrival_state["r"], arrival_state["v"], arrival_state["t"]
    tof = t2 - t1
    if tof <= 0:
        raise ValueError("Arrival epoch must be after departure epoch.")
    if (thrust is None) != (mass is None):
        raise ValueError("thrust and mass must be supplied together.")
    if burn_accel is not None and thrust is not None:
        raise ValueError(
            "Specify either burn_accel or thrust+mass, not both.")
    if isp is not None and mass is None:
        raise ValueError(
            "isp requires mass and thrust (propellant estimates are "
            "unavailable with burn_accel alone).")
    a_spec = (thrust / mass) if thrust is not None else burn_accel

    # --- Lambert solution and burns --------------------------------------
    v1_req, v2_req = solve_lambert(r1, r2, tof, mu=mu, prograde=prograde)

    dv1 = v1_req - v1          # departure burn
    dv2 = v2 - v2_req          # arrival (matching) burn

    # NTW components relative to the local state at each burn.
    dv1_ntw = rv_to_ntw(r1, v1, r1 + dv1) - rv_to_ntw(r1, v1, r1)
    dv2_ntw = rv_to_ntw(r2, v2_req, r2 + dv2) - rv_to_ntw(r2, v2_req, r2)

    # Burn durations: hardware-sized (constant accel thrust/mass) or the
    # requested burn_duration.  A small floor keeps near-zero burns
    # numerically sane.
    if a_spec is not None:
        d1 = max(np.linalg.norm(dv1) / a_spec, 1e-3)
        d2 = (max(np.linalg.norm(dv2) / a_spec, 1e-3)
              if arrival_burn else 0.0)
    else:
        d1 = burn_duration
        d2 = burn_duration if arrival_burn else 0.0
    if d1 + d2 >= tof / 3.0:
        raise ValueError(
            f"Burn durations {d1:.1f} s + {d2:.1f} s exceed a third of "
            f"the time of flight ({tof:.1f} s). Beyond that the "
            "constant-NTW finite-burn model is unreliable (large "
            "steering/gravity losses and poor refinement convergence). "
            + ("The engine (accel "
               f"{a_spec:.4f} m/s^2) is too weak for this "
               "transfer; increase the burn acceleration or allow a "
               "longer time of flight." if a_spec is not None else
               "Reduce burn_duration."))

    burns = [Burn(t1, t1 + d1, dv1, dv1_ntw)]
    if arrival_burn:
        burns.append(Burn(t2 - d2, t2, dv2, dv2_ntw))

    transfer_orbit = Orbit(r1, v1_req, t1, mu=mu)

    # --- optional differential correction (shooting) ----------------------
    if propagate:
        from ssapy.compute import rv

        orbit0 = Orbit(r1, v1, t1, mu=mu)

        # Propagate piecewise -- burn 1 / coast / burn 2 -- with each
        # segment's force model smooth over its whole span (the burn
        # accelerations use always-on AccelConstNTW within their
        # segment). Stepping an adaptive integrator across the on/off
        # discontinuity of a breakpointed burn produces erratic,
        # tolerance-sensitive errors that wreck shooting Jacobians;
        # aligning segment boundaries with the discontinuities removes
        # the problem at any epoch with default tolerances.
        # Outer duration-sizing loop (hardware mode): burn durations are
        # sized from the delta-v, but the refined delta-v exceeds the
        # impulsive estimate (finite-burn steering losses), so iterate
        # size -> refine until each burn's implied thrust m*|dv|/d
        # matches the specified engine to ~2%.
        a_burn = a_spec
        for _sizing in range(4):
            if arrival_burn:
                seg_edges = [t1, t1 + d1, t2 - d2, t2]
            else:
                seg_edges = [t1, t1 + d1, t2]

            def make_propagator(seg_accel, span):
                """Instantiate the per-segment propagator."""
                if propagator is None:
                    return RK78Propagator(seg_accel, h=min(rk_step, span))
                if not callable(propagator):
                    raise TypeError(
                        "propagator must be a propagator class or a factory "
                        "callable f(accel) -> propagator, not an instance: "
                        "each transfer segment is propagated under a "
                        "different acceleration sum.")
                if isinstance(propagator, type):
                    try:
                        return propagator(seg_accel, h=min(rk_step, span))
                    except TypeError:
                        return propagator(seg_accel)
                return propagator(seg_accel)

            def propagate_piecewise(dv_ntw_pair, sample_times=None):
                """Propagate the three segments; return final (r, v) and,
                if ``sample_times`` is given, the sampled trajectory."""
                seg_accels = [
                    accel + AccelConstNTW(dv_ntw_pair[:3] / d1),
                    accel,
                ]
                if arrival_burn:
                    seg_accels.append(
                        accel + AccelConstNTW(dv_ntw_pair[3:] / d2))
                state = orbit0
                traj_r, traj_v, traj_t = [], [], []
                if sample_times is not None:
                    # Serve the initial epoch from the known state directly.
                    # Querying a propagator at (or before) its orbit epoch
                    # triggers a backward integration that perturbs the
                    # interpolating spline, shifting the served segment
                    # endpoint -- which must stay identical between the
                    # refinement (endpoint-only) and trajectory-sampled
                    # propagations.
                    traj_t.append(t1)
                    traj_r.append(np.asarray(orbit0.r, dtype=float))
                    traj_v.append(np.asarray(orbit0.v, dtype=float))
                for (ta, tb), seg_accel in zip(
                        zip(seg_edges[:-1], seg_edges[1:]), seg_accels):
                    prop = make_propagator(seg_accel, tb - ta)
                    ts = [tb]
                    if sample_times is not None:
                        ts = [t for t in sample_times if ta < t < tb] + ts
                    r_, v_ = rv(state, np.array(ts), propagator=prop)
                    r_, v_ = np.atleast_2d(r_), np.atleast_2d(v_)
                    if r_.shape[0] < len(ts):
                        raise RuntimeError(
                            "Propagation terminated early (SSAPy detected an "
                            "Earth impact): the transfer arc dips below the "
                            "Earth's surface. Choose a longer time of flight "
                            "or different boundary geometry; the Keplerian "
                            "transfer conic's perigee can be checked via "
                            "result.transfer_orbit with propagate=False.")
                    state = Orbit(r_[-1], v_[-1], tb, mu=mu)
                    if sample_times is not None:
                        traj_t.extend(ts[:-1])
                        traj_r.extend(r_[:-1])
                        traj_v.extend(v_[:-1])
                if sample_times is not None:
                    traj_t.append(seg_edges[-1])
                    traj_r.append(r_[-1])
                    traj_v.append(v_[-1])
                    return (np.ravel(r_[-1]), np.ravel(v_[-1]),
                            np.array(traj_t), np.array(traj_r),
                            np.array(traj_v))
                return np.ravel(r_[-1]), np.ravel(v_[-1])

            def end_state(dv_ntw_pair):
                rf, vf = propagate_piecewise(dv_ntw_pair)
                return rf, vf

            x = (np.concatenate([dv1_ntw, dv2_ntw]) if arrival_burn
                 else np.asarray(dv1_ntw, dtype=float).copy())
            if refine:
                pos_tol, vel_tol = refine_tol

                # Two-stage targeting. Burn 1 controls the arrival *position*
                # (a long lever arm over the whole coast); burn 2 controls the
                # arrival *velocity* (its effect on position over its short
                # duration is negligible). Solving the two 3x3 problems
                # alternately is far better conditioned than a joint 6x6
                # solve, whose burn-2/position block is integrator noise.
                def newton_3x3(x, sl, want, get, tol):
                    """Damped Newton on x[sl] driving get(state)->want.

                    On a stalled step (no improving damped step found)
                    the finite-difference step is refined and the
                    iteration retried -- long burns make the map
                    nonlinear enough that a coarse Jacobian can stall
                    short of tolerance.
                    """
                    fd = 1e-2
                    rf, vf = end_state(x)
                    resid = get(rf, vf) - want
                    for _ in range(2 * max_refine_iter):
                        if np.linalg.norm(resid) < tol:
                            break
                        J = np.empty((3, 3))
                        for j in range(3):
                            dx = np.zeros(x.size)
                            dx[sl.start + j] = fd
                            rp, vp = end_state(x + dx)
                            J[:, j] = (get(rp, vp) - get(rf, vf)) / fd
                        if np.max(np.abs(J)) < 1e-12:
                            raise ValueError(
                                "The propagated end state does not respond to "
                                "the burn accelerations; the chosen propagator "
                                "is likely analytic (e.g. KeplerianPropagator, "
                                "SGP4Propagator, SeriesPropagator) and ignores "
                                "supplied accels. Use a numerical propagator "
                                "such as RK78Propagator or RK4Propagator.")
                        try:
                            step = np.linalg.solve(J, resid)
                        except np.linalg.LinAlgError:
                            step = np.linalg.lstsq(J, resid, rcond=None)[0]
                        # Clamp: never command more than a 100 m/s correction
                        # in one iteration (guards against noise-driven steps
                        # whose huge accelerations stall the integrator).
                        sn = np.linalg.norm(step)
                        if sn > 100.0:
                            step *= 100.0 / sn
                        best = np.linalg.norm(resid)
                        alpha = 1.0
                        for _ in range(8):
                            x_try = x.copy()
                            x_try[sl] = x[sl] - alpha * step
                            rt, vt = end_state(x_try)
                            resid_try = get(rt, vt) - want
                            if np.linalg.norm(resid_try) < best:
                                x, rf, vf, resid = x_try, rt, vt, resid_try
                                break
                            alpha *= 0.5
                        else:
                            if fd > 2e-4:
                                fd *= 0.1   # finer Jacobian, try again
                                continue
                            break  # no improving step; keep best so far
                    return x, np.linalg.norm(resid)

                if not arrival_burn:
                    # Intercept: a single position-targeting stage.
                    x, pos_err = newton_3x3(
                        x, slice(0, 3), r2, lambda rr, vv: rr, pos_tol)
                else:
                    for _ in range(max_refine_iter):
                        x, pos_err = newton_3x3(
                            x, slice(0, 3), r2, lambda rr, vv: rr, pos_tol)
                        x, vel_err = newton_3x3(
                            x, slice(3, 6), v2, lambda rr, vv: vv, vel_tol)
                        if pos_err < pos_tol and vel_err < vel_tol:
                            # Burn 2's correction nudges position slightly;
                            # re-check before declaring victory.
                            rf, vf = end_state(x)
                            if (np.linalg.norm(rf - r2) < pos_tol
                                    and np.linalg.norm(vf - v2) < vel_tol):
                                break


            if a_burn is None:
                break
            nd1 = max(np.linalg.norm(x[:3]) / a_burn, 1e-3)
            nd2 = (max(np.linalg.norm(x[3:]) / a_burn, 1e-3)
                   if arrival_burn else 0.0)
            if nd1 + nd2 >= tof / 3.0:
                raise ValueError(
                    f"Once finite-burn steering losses are included, the "
                    f"burn durations ({nd1:.1f} s + {nd2:.1f} s) exceed "
                    f"a third of the time of flight ({tof:.1f} s); the "
                    f"engine (accel {a_burn:.4f} m/s^2) is too weak for this "
                    "transfer. Increase thrust, reduce mass, or allow a "
                    "longer time of flight.")
            drift = abs(nd1 - d1) / d1
            if arrival_burn:
                drift = max(drift, abs(nd2 - d2) / max(d2, 1e-3))
            if drift < 0.02:
                break               # durations consistent with refined dv
            d1, d2 = nd1, nd2
            # Warm-start the next refinement from this solution.
            dv1_ntw = x[:3].copy()
            if arrival_burn:
                dv2_ntw = x[3:].copy()

        # Final piecewise propagation with the (possibly refined) burns,
        # sampled for the returned trajectory.
        sample_times = np.linspace(t1, t2, n_samples)
        rf, vf, tt, rr_traj, vv_traj = propagate_piecewise(
            x, sample_times=sample_times)
        burns = [Burn(t1, t1 + d1,
                      _ntw_to_inertial(r1, v1, x[:3]), x[:3])]
        if arrival_burn:
            burns.append(Burn(t2 - d2, t2,
                              _ntw_to_inertial(r2, v2_req, x[3:]), x[3:]))
        result = TransferResult(burns, dv_budget, transfer_orbit)
        result.trajectory = {"t": tt, "r": rr_traj, "v": vv_traj}
        result.arrival_error = float(np.linalg.norm(rf - r2))
        # Convenience propagator: full force model plus both breakpointed
        # burns, suitable for ssapy.compute.rv from the pre-burn state.
        full_accel = accel + burns[0].accel
        if arrival_burn:
            full_accel = full_accel + burns[1].accel
        result.propagator = make_propagator(full_accel, tof)
    else:
        result = TransferResult(burns, dv_budget, transfer_orbit)

    _attach_engine_info(result.burns, thrust, mass, isp)

    if dv_budget is not None and not result.within_budget:
        msg = (f"Transfer requires {result.dv_total:.3f} m/s, exceeding "
               f"the {dv_budget:.3f} m/s budget.")
        if raise_on_budget:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)

    return _transfer_ssapy_to_standard_dict(
        result,
        r1=r1,
        v1=v1,
        t1=t1,
        r2=r2,
        v2=v2,
        t2=t2,
        v2_req=v2_req,
        tof=tof,
        mu=mu,
        prograde=prograde,
        arrival_mode=arrival_mode,
        arrival_burn=arrival_burn,
        propagate=propagate,
        refine=refine,
        burn_accel=burn_accel,
        thrust=thrust,
        mass=mass,
        isp=isp,
    )


def _transfer_ssapy_to_standard_dict(
    result,
    *,
    r1,
    v1,
    t1,
    r2,
    v2,
    t2,
    v2_req,
    tof,
    mu,
    prograde,
    arrival_mode,
    arrival_burn,
    propagate,
    refine,
    burn_accel,
    thrust,
    mass,
    isp,
):
    initial_state = {"r": r1, "v": v1, "t": t1}
    target_state = {"r": r2, "v": v2, "t": t2}
    if result.trajectory is not None:
        final_state = {
            "r": result.trajectory["r"][-1],
            "v": result.trajectory["v"][-1],
            "t": result.trajectory["t"][-1],
        }
        trajectory = trajectory_dict(
            t=result.trajectory["t"],
            r=result.trajectory["r"],
            v=result.trajectory["v"],
        )
    else:
        final_state = target_state
        trajectory = None

    burn_states = [initial_state]
    if len(result.burns) > 1:
        burn_states.append({"r": r2, "v": v2_req, "t": t2})
    burns = []
    for index, burn in enumerate(result.burns):
        state = burn_states[index]
        burn_dict = maneuver_burn(
            name="departure" if index == 0 else "arrival_match",
            kind="constant_ntw_finite_burn",
            state=state,
            delta_v=burn.dv,
            t=burn.t_start,
            t_start=burn.t_start,
            t_end=burn.t_end,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass if thrust is not None else None,
            isp=isp,
        )
        if getattr(burn, "propellant_mass", None) is not None:
            burn_dict["propellant_mass"] = burn.propellant_mass
        burn_dict["delta_v_ntw"] = burn.dv_ntw
        burn_dict["acceleration_ntw"] = burn.dv_ntw / max(burn.t_end - burn.t_start, 1e-12)
        burns.append(burn_dict)

    return transfer_result(
        method="transfer_ssapy",
        initial=initial_state,
        target=target_state,
        final=final_state,
        burns=burns,
        trajectory=trajectory,
        transfer_orbits=[result.transfer_orbit],
        tof=tof,
        success=(result.arrival_error is None or result.arrival_error < np.inf),
        assumptions=[
            "zero-revolution Lambert solution",
            "two-body Lambert geometry",
            "constant NTW finite burns for propagation/refinement",
        ],
        diagnostics={
            "arrival_error": result.arrival_error,
            "dv_budget": result.dv_budget,
            "within_budget": result.within_budget,
            "prograde": bool(prograde),
            "arrival_mode": arrival_mode,
            "timing_constraint": "fixed",
            "arrival_velocity_match": bool(arrival_burn),
            "arrival_burn": bool(arrival_burn),
            "propagate": bool(propagate),
            "refine": bool(refine),
            "mu": mu,
        },
    )


def _attach_engine_info(burns, thrust, mass, isp):
    """Attach hardware context and Tsiolkovsky propellant estimates.

    Reporting only -- the propagated dynamics ignore mass depletion.
    The running mass is debited between burns purely for the estimate.
    """
    m = mass
    for b in burns:
        b.thrust = thrust
        b.duration = b.t_end - b.t_start
        b.propellant_mass = None
        if isp is not None and m is not None:
            b.propellant_mass = m * (1.0 - np.exp(-b.dv_mag / (isp * G0)))
            m = m - b.propellant_mass


def _ntw_to_inertial(r, v, ntw):
    """Rotate an NTW-frame vector into the inertial frame at state (r, v)."""
    tvec = normed(np.asarray(v, dtype=float))
    wvec = normed(np.cross(r, v))
    nvec = normed(np.cross(tvec, wvec))
    return ntw[0] * nvec + ntw[1] * tvec + ntw[2] * wvec
