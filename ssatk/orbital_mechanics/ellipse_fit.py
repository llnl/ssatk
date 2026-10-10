#!/usr/bin/env python3
"""
ellipse_fit.py  ────────────────────────────────────────────────────────────────
Generate sample points and analytic velocities along a Keplerian ellipse in ℝ³
whose first focus is at the origin.

Direction selection
-------------------
Instead of specifying CCW/CW, you can pass a preferred inertial velocity vector
`v_pref_m_s` (3,) at P1. The function will consider BOTH motion directions in the
same geometric plane (w_hat and -w_hat) and choose the one whose initial velocity
direction best matches `v_pref_m_s` (max cosine similarity). If `v_pref_m_s` is None,
the shorter time-of-flight direction is chosen.

Sampling
--------
Primary path: compute r, v, t_rel using Kepler's equation (mean anomaly) so that
a pure 2-body propagator should match closely.
Fallback: if Kepler sampling fails for any reason, fall back to the previous
polygon/angle-integral method.

Inclination
-----------
- `inc` is interpreted in RADIANS.
- For backward compatibility, you may pass `inc_deg=<degrees>` instead.

Returns
-------
dict
    (same keys as before)
"""

from ..plots import figsave
from ..plots.plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs
import numpy as np
import warnings


def ellipse_fit(
    P1_m,
    P2_m,
    *,
    a_m=None,
    e=None,
    F2_m=None,
    inc: float = 0.0,
    inc_deg=None,
    n_pts: int = 1000,
    tol=1e-8,
    v_pref_m_s=None,  # preferred velocity direction at P1 (in inertial frame)
    plot=False,
    save_path=False,
    time_of_departure=None,
    time_of_arrival=None,
    **save_kwargs,
):
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "ellipse_fit")

    from ..constants import EARTH_MU, EARTH_RADIUS  # [m³ s⁻²], [m]

    mu_m3_s2 = EARTH_MU
    R_earth_m = EARTH_RADIUS

    # interpret inclination
    if inc_deg is not None:
        incl_rad = np.deg2rad(float(inc_deg))
    else:
        incl_rad = float(inc)

    # ───────────────────────── internal helpers ──────────────────────────
    def _plane_basis(p1_m, p2_m, *, incl_rad: float = 0.0, eps: float = 1e-12):
        p1_m = np.asarray(p1_m, float)
        p2_m = np.asarray(p2_m, float)
        if np.linalg.norm(p1_m) < eps or np.linalg.norm(p2_m) < eps:
            raise ValueError("P1 and P2 must be non-zero vectors.")

        u_hat = p1_m / np.linalg.norm(p1_m)
        v_tmp = p2_m - np.dot(p2_m, u_hat) * u_hat

        if np.linalg.norm(v_tmp) < eps:
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(u_hat, cand)) > 1.0 - eps:
                cand = np.array([1.0, 0.0, 0.0])

            v0_hat = np.cross(u_hat, cand)
            v0_hat /= np.linalg.norm(v0_hat)

            w0_hat = np.cross(u_hat, v0_hat)
            w0_hat /= np.linalg.norm(w0_hat)

            theta_rad = incl_rad
            v_hat = np.cos(theta_rad) * v0_hat + np.sin(theta_rad) * w0_hat

            w_hat = np.cross(u_hat, v_hat)
            w_hat /= np.linalg.norm(w_hat)
            v_hat = np.cross(w_hat, u_hat)
            v_hat /= np.linalg.norm(v_hat)
        else:
            w_hat = np.cross(u_hat, v_tmp)
            w_hat /= np.linalg.norm(w_hat)
            v_hat = np.cross(w_hat, u_hat)
            v_hat /= np.linalg.norm(v_hat)

        return u_hat, v_hat, w_hat

    def _in_plane_m(vec3_m, u_hat, v_hat):
        return np.array([np.dot(vec3_m, u_hat), np.dot(vec3_m, v_hat)])

    def _to_3d_m(xy_m, u_hat, v_hat):
        return xy_m[0] * u_hat + xy_m[1] * v_hat

    def _eccentricity_from_focus(f2_m, a_m_):
        return np.linalg.norm(f2_m) / (2.0 * a_m_)

    def _a_for_focus(f2_m, p_m):
        return 0.5 * (np.linalg.norm(p_m) + np.linalg.norm(p_m - f2_m))

    def _solve_focus(P1_m_, P2_m_, u_hat, v_hat, *, a_m=None, e=None, F2_m=None, tol=1e-10):
        from scipy.optimize import root, minimize, NonlinearConstraint, BFGS

        p1_2d_m = _in_plane_m(P1_m_, u_hat, v_hat)
        p2_2d_m = _in_plane_m(P2_m_, u_hat, v_hat)

        if F2_m is not None:
            if (a_m is not None) or (e is not None):
                raise ValueError("Specify only one of a_m, e or F2_m.")
            F2_m = np.asarray(F2_m, float)
            a_m_ = _a_for_focus(F2_m, P1_m_)
            e_ = _eccentricity_from_focus(F2_m, a_m_)
            # Foci at the origin and F2 fix the ellipse through P1; P2 must lie
            # on it too (|P2| + |P2 - F2| = 2a) and F2 must lie in their plane.
            w_unit = np.cross(u_hat, v_hat)
            off_plane_m = abs(float(np.dot(F2_m, w_unit)))
            sum_residual_m = np.linalg.norm(P2_m_) + np.linalg.norm(P2_m_ - F2_m) - 2.0 * a_m_
            scale_m = max(np.linalg.norm(P1_m_), np.linalg.norm(P2_m_))
            if off_plane_m > 1e-6 * scale_m or abs(sum_residual_m) > 1e-6 * scale_m:
                raise ValueError(
                    "F2_m does not define an ellipse through both P1 and P2: "
                    f"|P2| + |P2 - F2| - 2a = {sum_residual_m:.3f} m, "
                    f"F2 is {off_plane_m:.3f} m out of the P1-P2 plane."
                )
            return F2_m, a_m_, e_

        if (a_m is None) and (e is None):

            def obj_min_e(xy_m):
                F_m = _to_3d_m(xy_m, u_hat, v_hat)
                return _eccentricity_from_focus(F_m, _a_for_focus(F_m, P1_m_))

            def equal_sum(xy_m):
                F_m = _to_3d_m(xy_m, u_hat, v_hat)
                return (
                    np.linalg.norm(P1_m_) + np.linalg.norm(P1_m_ - F_m)
                    - np.linalg.norm(P2_m_) - np.linalg.norm(P2_m_ - F_m)
                )

            midpoint = 0.5 * (p1_2d_m + p2_2d_m)
            chord = p2_2d_m - p1_2d_m
            chord_norm = np.linalg.norm(chord)

            if chord_norm > 0:
                perp = np.array([-chord[1], chord[0]]) / chord_norm
            else:
                perp = np.array([0.0, 1.0])

            scale = max(np.linalg.norm(p1_2d_m), np.linalg.norm(p2_2d_m), 1.0)

            guesses = [
                midpoint,
                np.zeros(2),
                0.25 * (p1_2d_m + p2_2d_m),
                0.75 * (p1_2d_m + p2_2d_m),
                midpoint + 0.25 * scale * perp,
                midpoint - 0.25 * scale * perp,
                midpoint + 0.50 * scale * perp,
                midpoint - 0.50 * scale * perp,
            ]

            constraint = NonlinearConstraint(equal_sum, 0.0, 0.0, jac="2-point", hess=BFGS())

            best_sol = None
            best_obj = np.inf
            debug_attempts = []

            for x0 in guesses:
                for method in ("trust-constr", "SLSQP"):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="delta_grad == 0.0.*",
                                category=UserWarning,
                            )
                            warnings.filterwarnings(
                                "ignore",
                                message="Singular Jacobian matrix.*",
                                category=UserWarning,
                            )
                            sol = minimize(
                                obj_min_e,
                                x0,
                                constraints=[constraint],
                                method=method,
                                tol=tol,
                            )
                        eq_res = abs(float(equal_sum(sol.x))) if hasattr(sol, "x") else np.inf
                        obj_val = float(obj_min_e(sol.x)) if hasattr(sol, "x") else np.inf
                        debug_attempts.append(
                            {
                                "method": method,
                                "x0": np.array(x0, float),
                                "success": bool(sol.success),
                                "status": getattr(sol, "status", None),
                                "message": str(getattr(sol, "message", "")),
                                "eq_res": eq_res,
                                "obj": obj_val,
                            }
                        )

                        if sol.success and np.isfinite(obj_val) and eq_res < max(tol, 1e-7):
                            if obj_val < best_obj:
                                best_sol = sol
                                best_obj = obj_val
                    except Exception as ex:
                        debug_attempts.append(
                            {
                                "method": method,
                                "x0": np.array(x0, float),
                                "success": False,
                                "status": None,
                                "message": f"EXCEPTION: {ex}",
                                "eq_res": np.inf,
                                "obj": np.inf,
                            }
                        )

            if best_sol is None:
                print("\n[ellipse_fit DEBUG] _solve_focus failed")
                print("P1_m_ =", P1_m_)
                print("P2_m_ =", P2_m_)
                print("p1_2d_m =", p1_2d_m)
                print("p2_2d_m =", p2_2d_m)
                print("midpoint =", midpoint)
                print("tol =", tol)
                for i, att in enumerate(debug_attempts):
                    print(
                        f"  attempt {i:02d}: method={att['method']}, success={att['success']}, "
                        f"status={att['status']}, eq_res={att['eq_res']:.3e}, obj={att['obj']:.6e}, "
                        f"message={att['message']}"
                    )
                raise RuntimeError("Could not locate least-eccentric solution.")

            F2_m_ = _to_3d_m(best_sol.x, u_hat, v_hat)
            a_m_ = _a_for_focus(F2_m_, P1_m_)
            e_ = _eccentricity_from_focus(F2_m_, a_m_)
            return F2_m_, a_m_, e_

        if (a_m is None) == (e is None):
            raise ValueError("Provide exactly one of {a_m, e, F2_m}.")

        if e is not None and not (0.0 < float(e) < 1.0):
            raise ValueError("e must satisfy 0 < e < 1 for ellipse fitting.")

        def residual(xy_m):
            F_m = _to_3d_m(xy_m, u_hat, v_hat)
            s1_m = np.linalg.norm(P1_m_) + np.linalg.norm(P1_m_ - F_m)
            s2_m = np.linalg.norm(P2_m_) + np.linalg.norm(P2_m_ - F_m)
            r1_m = s1_m - s2_m
            target_m = (np.linalg.norm(F_m) / e) if (e is not None) else (2.0 * a_m)
            r2_m = s1_m - target_m
            return np.array([r1_m, r2_m])

        sol = root(residual, 0.3 * p1_2d_m, tol=tol)
        if not sol.success:
            raise RuntimeError("Could not satisfy the supplied (a_m, e).")

        F2_m_ = _to_3d_m(sol.x, u_hat, v_hat)
        if e is None:
            e_ = _eccentricity_from_focus(F2_m_, a_m)
            a_m_ = a_m
        else:
            e_ = float(e)
            a_m_ = np.linalg.norm(F2_m_) / (2.0 * e_)
        return F2_m_, a_m_, e_

    def _unit(vec, eps=1e-15):
        vec = np.asarray(vec, float)
        nrm = np.linalg.norm(vec)
        if nrm < eps:
            return None
        return vec / nrm

    def _wrap_0_2pi(x_rad):
        x = np.asarray(x_rad, float)
        y = np.mod(x, 2.0 * np.pi)
        return y

    def _f_from_r(r_m, p_hat, w_hat):
        r_m = np.asarray(r_m, float)
        r_hat = r_m / np.linalg.norm(r_m)
        cos_f = float(np.dot(p_hat, r_hat))
        sin_f = float(np.dot(w_hat, np.cross(p_hat, r_hat)))
        return float(np.arctan2(sin_f, cos_f))

    def _E_from_f(f_rad, e_):
        # robust conversion true anomaly -> eccentric anomaly
        # tan(E/2) = sqrt((1-e)/(1+e)) * tan(f/2)
        beta = np.sqrt(max((1.0 - e_) / (1.0 + e_), 0.0))
        t = np.tan(0.5 * f_rad)
        E = 2.0 * np.arctan2(beta * t, 1.0)
        return float(E)

    def _M_from_E(E_rad, e_):
        return float(E_rad - e_ * np.sin(E_rad))

    def _solve_kepler_E(M_rad, e_, max_iter=50, tol=1e-12):
        """
        Vectorized Newton solve for E - e sinE = M.
        Returns E with same shape as M_rad.
        """
        M = np.asarray(M_rad, float)

        # initial guess
        E = M.copy()
        if e_ > 0.8:
            # slightly better guess for high-e
            E = np.pi * np.ones_like(M)
            E = np.where(M < 0.0, -np.pi, E)
        else:
            E = M + e_ * np.sin(M)

        for _ in range(int(max_iter)):
            f = E - e_ * np.sin(E) - M
            fp = 1.0 - e_ * np.cos(E)
            dE = -f / fp
            E = E + dE
            if np.max(np.abs(dE)) < tol:
                break
        return E

    def _rv_from_M_grid(M_grid_rad, *, a_m, e_, mu_m3_s2, p_hat, w_hat):
        """
        Build r,v in 3D from mean anomaly grid.

        Uses p_hat (periapsis direction) and w_hat (orbit normal) to define
        q_hat = w_hat × p_hat.
        """
        p_hat_u = _unit(p_hat)
        w_hat_u = _unit(w_hat)
        if p_hat_u is None or w_hat_u is None:
            raise ValueError("Invalid p_hat or w_hat in _rv_from_M_grid.")

        q_hat = np.cross(w_hat_u, p_hat_u)
        q_hat_u = _unit(q_hat)
        if q_hat_u is None:
            raise ValueError("Degenerate q_hat in _rv_from_M_grid.")

        n_rad_s = np.sqrt(mu_m3_s2 / (a_m ** 3))

        E_rad = _solve_kepler_E(M_grid_rad, e_)
        cosE = np.cos(E_rad)
        sinE = np.sin(E_rad)

        # position in PQW from eccentric anomaly
        x_m = a_m * (cosE - e_)
        y_m = a_m * (np.sqrt(max(1.0 - e_ ** 2, 0.0)) * sinE)
        r_m = np.sqrt(x_m ** 2 + y_m ** 2)

        # velocity in PQW
        denom = (1.0 - e_ * cosE)
        xdot_m_s = (-a_m * sinE) * (n_rad_s / denom)
        ydot_m_s = (a_m * np.sqrt(max(1.0 - e_ ** 2, 0.0)) * cosE) * (n_rad_s / denom)

        r3_m = (x_m[:, None] * p_hat_u[None, :]) + (y_m[:, None] * q_hat_u[None, :])
        v3_m_s = (xdot_m_s[:, None] * p_hat_u[None, :]) + (ydot_m_s[:, None] * q_hat_u[None, :])

        return r3_m, v3_m_s, E_rad

    # ─────────────────────────── inputs / guards ──────────────────────────
    if (time_of_departure is not None) and (time_of_arrival is not None):
        raise ValueError("Specify either time_of_departure or time_of_arrival, not both.")

    P1_m = np.asarray(P1_m, float)
    P2_m = np.asarray(P2_m, float)

    u_hat, v_hat, w_hat = _plane_basis(P1_m, P2_m, incl_rad=incl_rad)

    F2_m, a_m, e = _solve_focus(P1_m, P2_m, u_hat, v_hat, a_m=a_m, e=e, F2_m=F2_m, tol=tol)

    # Secondary geometry scalars (still useful outputs)
    c_m = 0.5 * np.linalg.norm(F2_m)
    b_m = np.sqrt(max(a_m * a_m - c_m * c_m, 0.0))

    # If F2≈0 (near-circular), pick periapsis direction as P1 direction
    if np.linalg.norm(F2_m) < 1e-12:
        p_hat = _unit(P1_m)
        if p_hat is None:
            p_hat = np.array([1.0, 0.0, 0.0])
    else:
        p_hat = -F2_m / np.linalg.norm(F2_m)  # points from focus to periapsis

    # ───────────────────────── primary: Kepler sampling ───────────────────
    # We'll try Kepler-based r,v,t_rel; if anything goes wrong, we fall back.
    kepler_ok = True
    kepler_error = None

    try:
        # Two candidate motion directions: w_hat and -w_hat
        cand = []
        for w_use in (w_hat, -w_hat):
            w_use_u = _unit(w_use)
            if w_use_u is None:
                continue

            f1_rad = _f_from_r(P1_m, p_hat, w_use_u)
            f2_rad = _f_from_r(P2_m, p_hat, w_use_u)

            # Forward-time in this convention corresponds to increasing f in [0,2pi)
            delta_f_rad = float(_wrap_0_2pi(f2_rad - f1_rad))

            # Convert endpoints to E, M; unwrap so M2 >= M1 along forward motion
            E1_rad = _E_from_f(f1_rad, e)
            E2_rad = _E_from_f(f2_rad, e)

            M1_rad = _M_from_E(E1_rad, e)
            M2_rad = _M_from_E(E2_rad, e)

            # unwrap M2 forward if needed (since forward delta_f is in [0,2pi))
            # For long arcs, M2 should be M2 + 2π if it otherwise lands behind.
            if delta_f_rad > 1e-15:
                if M2_rad <= M1_rad:
                    M2_rad += 2.0 * np.pi
            else:
                # essentially same point; keep tiny positive span
                if M2_rad <= M1_rad:
                    M2_rad = M1_rad + 1e-15

            n_rad_s = float(np.sqrt(mu_m3_s2 / (a_m ** 3)))
            T_flight_s = float((M2_rad - M1_rad) / n_rad_s)

            # Initial velocity direction for scoring (from the same convention)
            # Build a tiny grid at M1 only
            r0_try_m, v0_try_m_s, _ = _rv_from_M_grid(
                np.array([M1_rad], dtype=float),
                a_m=a_m, e_=e, mu_m3_s2=mu_m3_s2, p_hat=p_hat, w_hat=w_use_u
            )
            v0_try_m_s = v0_try_m_s[0]

            score = 0.0
            if v_pref_m_s is not None:
                vph = _unit(v_pref_m_s)
                v0h = _unit(v0_try_m_s)
                if vph is None or v0h is None:
                    score = -np.inf
                else:
                    score = float(np.dot(v0h, vph))

            cand.append({
                "w_use": w_use_u,
                "f1_rad": f1_rad,
                "f2_rad": f2_rad,
                "delta_f_rad": delta_f_rad,
                "E1_rad": E1_rad,
                "E2_rad": E2_rad,
                "M1_rad": M1_rad,
                "M2_rad": M2_rad,
                "T_flight_s": T_flight_s,
                "score": score,
                "v0_m_s": v0_try_m_s,
            })

        if len(cand) < 1:
            raise RuntimeError("No valid motion-direction candidates for Kepler sampling.")

        # choose candidate
        if v_pref_m_s is None:
            # choose shorter time-of-flight by default
            chosen = min(cand, key=lambda d: d["T_flight_s"])
        else:
            chosen = max(cand, key=lambda d: d["score"])

        w_use = chosen["w_use"]
        M1_rad = chosen["M1_rad"]
        M2_rad = chosen["M2_rad"]
        T_flight_s = chosen["T_flight_s"]

        # time grid (uniform in time)
        N = int(max(2, int(n_pts)))
        t_rel_s = np.linspace(0.0, T_flight_s, N)

        n_rad_s = float(np.sqrt(mu_m3_s2 / (a_m ** 3)))
        M_grid_rad = M1_rad + n_rad_s * t_rel_s

        # build r,v from Kepler
        arc3d_m, vel3d_m_s, _E_grid = _rv_from_M_grid(
            M_grid_rad,
            a_m=a_m, e_=e, mu_m3_s2=mu_m3_s2, p_hat=p_hat, w_hat=w_use
        )

        # rot_dir: +1 if CCW seen along +w_hat; depends on whether we flipped w.
        # We define rot_dir w.r.t. the *original* w_hat from plane_basis.
        rot_dir = 1 if float(np.dot(w_use, w_hat)) >= 0.0 else -1

        # For plotting companion arc, generate the opposite direction candidate if available
        arc3d_comp_m = None
        if len(cand) == 2:
            other = cand[0] if cand[1] is chosen else cand[1]
            w_other = other["w_use"]
            M1o = other["M1_rad"]
            M2o = other["M2_rad"]
            To = other["T_flight_s"]
            t_rel_o_s = np.linspace(0.0, To, N)
            Mo = M1o + n_rad_s * t_rel_o_s
            arc3d_comp_m, _, _ = _rv_from_M_grid(
                Mo, a_m=a_m, e_=e, mu_m3_s2=mu_m3_s2, p_hat=p_hat, w_hat=w_other
            )

    except Exception as ex:
        kepler_ok = False
        kepler_error = ex

    # ───────────────────────── fallback: previous method ──────────────────
    if not kepler_ok:
        # Fallback: the earlier polygon/angle-integral sampling, used only if
        # Kepler sampling raised (the exception is kept in kepler_error).

        # ---- previous “polygon ellipse + dt/df” method ----
        f2_2d_m = _in_plane_m(F2_m, u_hat, v_hat)
        C_2d_m = 0.5 * f2_2d_m

        phi_rad = -np.arctan2(C_2d_m[1], C_2d_m[0])
        R2D = np.array([[np.cos(phi_rad), -np.sin(phi_rad)], [np.sin(phi_rad), np.cos(phi_rad)]])

        t_full_rad = np.linspace(0.0, 2.0 * np.pi, int(max(4, n_pts)), endpoint=False)
        xy_full_m = np.vstack((a_m * np.cos(t_full_rad), b_m * np.sin(t_full_rad))).T
        xy_full_m = (R2D.T @ xy_full_m.T).T + C_2d_m
        full3d_m = np.array([_to_3d_m(p_m, u_hat, v_hat) for p_m in xy_full_m])

        i1 = int(np.argmin(np.linalg.norm(full3d_m - P1_m, axis=1)))
        i2 = int(np.argmin(np.linalg.norm(full3d_m - P2_m, axis=1)))

        def slice_arc_segments(full_m, i1, i2):
            n = len(full_m)
            if i2 >= i1:
                short_idx = np.arange(i1, i2 + 1)
            else:
                short_idx = np.concatenate((np.arange(i1, n), np.arange(0, i2 + 1)))

            all_idx = np.arange(n)
            mask = np.ones(n, dtype=bool)
            mask[short_idx] = False
            long_idx = all_idx[mask]
            long_idx = np.concatenate((long_idx, [i1]))

            short_arc_m = full_m[short_idx]
            long_arc_m = full_m[long_idx]
            return short_arc_m, long_arc_m

        def direction(arc_m, u_hat, v_hat, w_hat, eps=1e-12):
            if len(arc_m) < 2:
                return 0
            p0_m = arc_m[0]
            p1_m = arc_m[min(10, len(arc_m) - 1)]
            s = np.dot(w_hat, np.cross(p0_m, p1_m))
            if abs(s) > eps:
                return int(np.sign(s))
            p0_2d_m = np.array([np.dot(p0_m, u_hat), np.dot(p0_m, v_hat)])
            p1_2d_m = np.array([np.dot(p1_m, u_hat), np.dot(p1_m, v_hat)])
            det = p0_2d_m[0] * p1_2d_m[1] - p0_2d_m[1] * p1_2d_m[0]
            if abs(det) < eps:
                return 0
            return int(np.sign(det))

        def _v_at_r0_m_s(r0_m, F2_m, e, a_m, w_hat, *, vel_mod=1):
            r0_m = np.asarray(r0_m, float)
            r_m = np.linalg.norm(r0_m)
            r_hat = r0_m / r_m

            if np.linalg.norm(F2_m) < 1e-12:
                e_hat = r_hat
            else:
                e_hat = -F2_m / np.linalg.norm(F2_m)

            h_m2_s = np.sqrt(mu_m3_s2 * a_m * (1.0 - e**2))

            cos_f = np.dot(e_hat, r_hat)
            sin_f = np.dot(w_hat, np.cross(e_hat, r_hat))
            v_r_m_s = (mu_m3_s2 / h_m2_s) * e * sin_f
            v_t_m_s = (mu_m3_s2 / h_m2_s) * (1.0 + e * cos_f)

            t_hat = np.cross(w_hat, r_hat)
            t_hat_u = _unit(t_hat)
            if t_hat_u is None:
                t_hat_u = np.zeros(3)

            return (v_r_m_s * r_hat + v_t_m_s * t_hat_u) * vel_mod

        arc_short_m, arc_long_m = slice_arc_segments(full3d_m, i1, i2)
        arc_long_m = arc_long_m[::-1]

        def _score_arc(arc_m):
            rot_dir_ = direction(arc_m, u_hat, v_hat, w_hat)
            vel_mod_ = 1 if rot_dir_ >= 0 else -1
            v0_m_s = _v_at_r0_m_s(arc_m[0], F2_m, e, a_m, w_hat, vel_mod=vel_mod_)

            if v_pref_m_s is None:
                return 0.0, rot_dir_, vel_mod_, v0_m_s

            v_pref_hat = _unit(v_pref_m_s)
            v0_hat = _unit(v0_m_s)
            if (v_pref_hat is None) or (v0_hat is None):
                return -np.inf, rot_dir_, vel_mod_, v0_m_s
            return float(np.dot(v0_hat, v_pref_hat)), rot_dir_, vel_mod_, v0_m_s

        score_s, rot_s, velmod_s, _v0_s = _score_arc(arc_short_m)
        score_l, rot_l, velmod_l, _v0_l = _score_arc(arc_long_m)

        if v_pref_m_s is None:
            arc3d_m = arc_short_m
            arc3d_comp_m = arc_long_m
            rot_dir = rot_s
            vel_mod = velmod_s
        else:
            if score_l > score_s:
                arc3d_m = arc_long_m
                arc3d_comp_m = arc_short_m
                rot_dir = rot_l
                vel_mod = velmod_l
            else:
                arc3d_m = arc_short_m
                arc3d_comp_m = arc_long_m
                rot_dir = rot_s
                vel_mod = velmod_s

        xy_arc_m = np.array([_in_plane_m(p_m, u_hat, v_hat) for p_m in arc3d_m])
        f_rad = np.unwrap(np.arctan2(xy_arc_m[:, 1], xy_arc_m[:, 0]))

        r_norm_m = np.linalg.norm(arc3d_m, axis=1)
        h_scalar_m2_s = np.sqrt(mu_m3_s2 * a_m * (1.0 - e**2))

        df_rad = np.diff(f_rad)
        df_abs_rad = np.abs(df_rad)

        dt_df_mid_s = 0.5 * (r_norm_m[:-1] ** 2 + r_norm_m[1:] ** 2) / h_scalar_m2_s
        dt_s = dt_df_mid_s * df_abs_rad

        t_rel_s = np.zeros_like(f_rad)
        t_rel_s[1:] = np.cumsum(dt_s)
        t_rel_s -= t_rel_s[0]

        vel3d_m_s = np.empty_like(arc3d_m)
        for k, r_vec_m in enumerate(arc3d_m):
            vel3d_m_s[k] = _v_at_r0_m_s(r_vec_m, F2_m, e, a_m, w_hat, vel_mod=vel_mod)

        # ensure companion exists for plotting
        if arc3d_comp_m is None:
            arc3d_comp_m = arc3d_m.copy()

    # ───────────────────────── convenience state values ───────────────────
    r0_m, v0_m_s = arc3d_m[0], vel3d_m_s[0]

    # absolute times (optional)
    t_abs = None
    if time_of_departure is not None:
        from ..time_functions import to_gps
        t0_abs_s = np.asarray(to_gps(time_of_departure))
        t_abs = t0_abs_s + t_rel_s
    elif time_of_arrival is not None:
        from ..time_functions import to_gps
        t1_abs_s = np.asarray(to_gps(time_of_arrival))
        t_abs = t1_abs_s - (t_rel_s[-1] - t_rel_s)

    # ───────────────────────── elements / invariants ──────────────────────
    r0_norm_m = np.linalg.norm(r0_m)
    h_vec_m2_s = np.cross(r0_m, v0_m_s)
    h_m2_s = np.linalg.norm(h_vec_m2_s)

    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec_m2_s)
    n_norm = np.linalg.norm(n_vec)

    e_vec = (np.cross(v0_m_s, h_vec_m2_s) / mu_m3_s2) - (r0_m / r0_norm_m)
    e = float(np.linalg.norm(e_vec))

    i_rad = np.arccos(h_vec_m2_s[2] / h_m2_s)

    if n_norm != 0.0:
        raan_rad = np.arccos(n_vec[0] / n_norm)
        if n_vec[1] < 0.0:
            raan_rad = 2.0 * np.pi - raan_rad
    else:
        raan_rad = 0.0

    if n_norm != 0.0 and e != 0.0:
        pa_rad = np.arccos(np.dot(n_vec, e_vec) / (n_norm * e))
        if e_vec[2] < 0.0:
            pa_rad = 2.0 * np.pi - pa_rad
    else:
        pa_rad = 0.0

    if e != 0.0:
        ta_rad = np.arccos(np.dot(e_vec, r0_m) / (e * r0_norm_m))
        if np.dot(r0_m, v0_m_s) < 0.0:
            ta_rad = 2.0 * np.pi - ta_rad
    else:
        ta_rad = 0.0

    cosE = (e + np.cos(ta_rad)) / (1.0 + e * np.cos(ta_rad))
    sinE = (np.sin(ta_rad) * np.sqrt(1.0 - e**2)) / (1.0 + e * np.cos(ta_rad))
    E_rad = np.arctan2(sinE, cosE)
    if E_rad < 0.0:
        E_rad += 2.0 * np.pi
    M_rad = E_rad - e * np.sin(E_rad)
    L_rad = (raan_rad + pa_rad + M_rad) % (2.0 * np.pi)

    period_s = 2.0 * np.pi * np.sqrt(a_m**3 / mu_m3_s2)

    # ─────────────────────────── plotting  ──────────────────────────────
    if plot or save_path:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit

        t_minutes = t_rel_s / 60.0
        r_ss_m, v_ss_m_s, _t_ss = ssapy_orbit(r=r0_m, v=v0_m_s, t=t_rel_s)

        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 2, 2])
        ax3d = fig.add_subplot(gs[0], projection="3d")
        ax_dist = fig.add_subplot(gs[1])
        ax_speed = fig.add_subplot(gs[2])

        colors = plt.get_cmap("RdYlGn_r")(np.linspace(0, 1, len(arc3d_m)))

        if arc3d_comp_m is not None:
            ax3d.scatter(
                arc3d_comp_m[:, 0] / 1e3,
                arc3d_comp_m[:, 1] / 1e3,
                arc3d_comp_m[:, 2] / 1e3,
                c="black",
                s=10,
            )

        ax3d.scatter(
            arc3d_m[:, 0] / 1e3,
            arc3d_m[:, 1] / 1e3,
            arc3d_m[:, 2] / 1e3,
            c=colors,
            s=10,
        )
        ax3d.plot(
            r_ss_m[:, 0] / 1e3,
            r_ss_m[:, 1] / 1e3,
            r_ss_m[:, 2] / 1e3,
            lw=2.5,
            color="gray",
            alpha=0.6,
            label="ssapy",
        )
        ax3d.scatter(0, 0, 0, marker="x", s=80, color="black", label="Earth")
        ax3d.scatter(*(F2_m / 1e3), marker="x", s=80, color="gray", alpha=0.5, label="focus")
        ax3d.scatter(*(P1_m / 1e3), s=40, label="P1", color="green")
        ax3d.scatter(*(P2_m / 1e3), s=40, label="P2", color="red")
        ax3d.set_xlabel("X [km]")
        ax3d.set_ylabel("Y [km]")
        ax3d.set_zlabel("Z [km]")
        Rmax_km = np.linalg.norm(arc3d_m, axis=1).max() / 1e3
        ax3d.set_xlim((-Rmax_km, Rmax_km))
        ax3d.set_ylim((-Rmax_km, Rmax_km))
        ax3d.set_zlim((-Rmax_km, Rmax_km))
        ax3d.set_title("Kepler-sampled ellipse with ssapy overlay")
        ax3d.legend()

        ax_dist.plot(t_minutes, np.linalg.norm(arc3d_m, axis=1) / 1e3, lw=5, label="ellipse")
        ax_dist.plot(t_minutes, np.linalg.norm(r_ss_m, axis=1) / 1e3, lw=5, ls="--", color="red", label="ssapy")
        ax_dist.set_xlabel("Time from P1 [min]")
        ax_dist.set_ylabel("Distance [km]")
        ax_dist.set_title("Distance from origin")
        ax_dist.grid(True)
        ax_dist.legend()

        ax_speed.plot(t_minutes, np.linalg.norm(vel3d_m_s, axis=1) / 1e3, lw=5, label="ellipse")
        ax_speed.plot(t_minutes, np.linalg.norm(v_ss_m_s, axis=1) / 1e3, lw=5, ls="--", color="red", label="ssapy")
        ax_speed.set_xlabel("Time from P1 [min]")
        ax_speed.set_ylabel("Speed [km/s]")
        ax_speed.set_title("Speed over time")
        ax_speed.grid(True)
        ax_speed.legend()

        plt.tight_layout()
        plt.show()
        if save_path:
            figsave(fig, save_path)

    # ───────────────────────── output dict (keys unchanged) ───────────────
    result = {
        "r": arc3d_m,                         # (N,3) [m]
        "v": vel3d_m_s,                       # (N,3) [m/s]
        "t_rel": t_rel_s,                     # (N,)  [s]
        "t_abs": t_abs,                       # (N,)  [s] (or other, depending on to_gps)
        "a": a_m,                             # [m]
        "e": e,                               # [-]
        "i": i_rad,                           # [rad]
        "raan": raan_rad,                     # [rad]
        "pa": pa_rad,                         # [rad]
        "ta": ta_rad,                         # [rad]
        "L": L_rad,                           # [rad]
        "rp": a_m * (1.0 - e),                # [m]
        "ra": a_m * (1.0 + e),                # [m]
        "rp_alt": a_m * (1.0 - e) - R_earth_m,# [m]
        "ra_alt": a_m * (1.0 + e) - R_earth_m,# [m]
        "b": b_m,                             # [m]
        "p": a_m * (1.0 - e**2),              # [m]
        "mean_motion": np.sqrt(mu_m3_s2 / a_m**3),  # [rad/s]
        "eta": np.sqrt(1.0 - e**2),           # [-]
        "period": period_s,                   # [s]
        "h_vec": h_vec_m2_s,                  # [m^2/s]
        "h": h_m2_s,                          # [m^2/s]
        "Energy": -mu_m3_s2 / (2.0 * a_m),    # [J/kg]
        "e_vec": e_vec,                       # [-]
        "n_vec": n_vec,                       # [-]
        "r0": r0_m,                           # [m]
        "v0": v0_m_s,                         # [m/s]
        "F2": F2_m,                           # [m]
        "plane_basis": (u_hat, v_hat, w_hat),
        "rot_dir": int(rot_dir),
        "mu": mu_m3_s2,                       # [m^3/s^2]
    }

    return result

"""
delta_v_transfer.py ─────────────────────────────────────────────────────────────
Turn an `ellipse_fit` transfer arc plus the departure/arrival orbit velocities
into the two impulsive burns (Δv) of a two-burn transfer.

A Δv is the difference between the transfer-arc velocity at an endpoint and the
velocity of the orbit the spacecraft is actually leaving (at P1) or joining
(at P2):

    Δv₁ = v_transfer(P1) − v_depart(P1)     # burn that injects onto the arc
    Δv₂ = v_arrive(P2)   − v_transfer(P2)   # burn that matches the target orbit
    Δv_total = |Δv₁| + |Δv₂|

`ellipse_fit` already supplies the transfer-arc velocities (`fit["v"][0]` at P1
and `fit["v"][-1]` at P2); this wrapper supplies the boundary-orbit velocities
and does the differencing. There is no orbital mechanics here beyond the
subtraction — all four vectors must already live in the SAME Earth-centered
inertial frame (the frame `ellipse_fit` works in, with the focus at the origin).

Boundary states may be passed as plain (3,) velocity arrays or as `ssapy.Orbit`
objects. When an `Orbit` is passed, its `.v` is differenced and its `.r` is
checked against the matching arc endpoint, so a mis-propagated or wrong-frame
boundary state is caught loudly instead of silently producing a garbage burn.
The `Orbit` must be propagated to the endpoint epoch first (i.e. a scalar-time
state, not a trajectory) — see `fit["t_abs"]` for those epochs when `ellipse_fit`
was called with a departure/arrival time.

Sign convention: both `dv1` and `dv2` are the velocity *added* to the spacecraft
(command burns), so dv1 = v_transfer − v_depart and dv2 = v_arrive − v_transfer.
Magnitudes are direction-independent regardless.
"""


def delta_v_transfer(
    fit,
    depart,
    arrive,
    *,
    pos_tol_m=1.0e3,
    check_positions=True,
):
    # ───────────────────────── internal helpers ──────────────────────────
    def _state_from(obj, name):
        """
        Return (v_m_s, r_m_or_None) from either a (3,) velocity array or an
        ssapy.Orbit. r is None when only a bare velocity vector was supplied.
        Duck-typed on .r/.v so ssapy is never hard-imported here.
        """
        r_attr = getattr(obj, "r", None)
        v_attr = getattr(obj, "v", None)
        if v_attr is not None and r_attr is not None:
            v = np.asarray(v_attr, float).reshape(-1)
            r = np.asarray(r_attr, float).reshape(-1)
            if v.size != 3 or r.size != 3:
                raise ValueError(
                    f"{name}: Orbit must carry a single 3-vector state "
                    f"(got r shape {np.shape(r_attr)}, v shape {np.shape(v_attr)}). "
                    f"Propagate to the endpoint epoch and pass a scalar-time Orbit."
                )
            return v, r

        v = np.asarray(obj, float).reshape(-1)
        if v.size != 3:
            raise ValueError(
                f"{name}: expected a (3,) velocity vector (or ssapy.Orbit), "
                f"got shape {np.shape(obj)}."
            )
        return v, None

    # ───────────────────────── transfer endpoints ─────────────────────────
    v_arc = np.asarray(fit["v"], float)
    r_arc = np.asarray(fit["r"], float)
    if v_arc.ndim != 2 or v_arc.shape[0] < 2:
        raise ValueError("fit['v'] must be an (N>=2, 3) array from ellipse_fit.")

    v_t1 = v_arc[0]    # transfer-arc velocity at P1 (== fit['v0'])
    v_t2 = v_arc[-1]   # transfer-arc velocity at P2
    r_p1 = r_arc[0]    # P1
    r_p2 = r_arc[-1]   # P2

    # ───────────────────────── boundary velocities ────────────────────────
    v_dep, r_dep = _state_from(depart, "depart")
    v_arr, r_arr = _state_from(arrive, "arrive")

    # ───────────────────────── position consistency ───────────────────────
    res_p1 = None if r_dep is None else float(np.linalg.norm(r_dep - r_p1))
    res_p2 = None if r_arr is None else float(np.linalg.norm(r_arr - r_p2))

    if check_positions:
        if res_p1 is not None and res_p1 > pos_tol_m:
            raise ValueError(
                f"Departure state is {res_p1:.3e} m from the arc start P1 "
                f"(tol {pos_tol_m:g} m). Propagate the departure orbit to the P1 "
                f"epoch, confirm the frame matches ellipse_fit, or relax pos_tol_m."
            )
        if res_p2 is not None and res_p2 > pos_tol_m:
            raise ValueError(
                f"Arrival state is {res_p2:.3e} m from the arc end P2 "
                f"(tol {pos_tol_m:g} m). Propagate the arrival orbit to the P2 "
                f"epoch, confirm the frame matches ellipse_fit, or relax pos_tol_m."
            )

    # ─────────────────────────────── burns ────────────────────────────────
    dv1 = v_t1 - v_dep
    dv2 = v_arr - v_t2
    dv1_mag = float(np.linalg.norm(dv1))
    dv2_mag = float(np.linalg.norm(dv2))

    # ───────────────────────── output dict ────────────────────────────────
    return {
        "dv1": dv1,                     # (3,) [m/s]  departure burn (v_transfer_P1 − v_depart)
        "dv2": dv2,                     # (3,) [m/s]  arrival burn   (v_arrive − v_transfer_P2)
        "dv1_mag": dv1_mag,             # [m/s]
        "dv2_mag": dv2_mag,             # [m/s]
        "dv_total": dv1_mag + dv2_mag,  # [m/s]
        "v_t1": v_t1,                   # (3,) [m/s]  transfer velocity at P1
        "v_t2": v_t2,                   # (3,) [m/s]  transfer velocity at P2
        "v_depart": v_dep,              # (3,) [m/s]
        "v_arrive": v_arr,              # (3,) [m/s]
        "pos_residual_p1": res_p1,      # [m]  |r_depart − P1|, or None if no Orbit given
        "pos_residual_p2": res_p2,      # [m]  |r_arrive − P2|, or None if no Orbit given
    }
