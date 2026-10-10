from astropy.time import Time

from ssatk.time_functions.get_times import get_times



# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False
def demo_get_times(verbose=True):
    results = {}

    # Demo 1: forward from t0 (1 hour, 10 min steps)
    t0 = "2027-01-01 12:00:00"
    times_forward = get_times(duration=(1, "hour"), freq=(10, "min"), t0=t0)
    results["forward"] = times_forward
    if verbose:
        print("Demo 1: forward from t0 (1 hour, 10 min steps)")
        print("  t0 (first):", times_forward[0].iso)
        print("  t_end (last):", times_forward[-1].iso)
        print("  number of steps:", len(times_forward))
        print("  all times:")
        for t in times_forward:
            print("   ", t.iso)
        print()

    # Demo 2: end at tf (1 hour, 10 min steps)
    tf = "2027-01-01 12:00:00"
    times_end_at_tf = get_times(duration=(1, "hour"), freq=(10, "min"), tf=tf)
    results["end_at_tf"] = times_end_at_tf
    if verbose:
        print("Demo 2: end at tf (1 hour, 10 min steps)")
        print("  t_start (first):", times_end_at_tf[0].iso)
        print("  tf (last):", times_end_at_tf[-1].iso)
        print("  number of steps:", len(times_end_at_tf))
        print("  all times:")
        for t in times_end_at_tf:
            print("   ", t.iso)
        print()

    # Demo 3: zero duration (single time)
    times_single = get_times(duration=0, freq=(10, "min"), t0=t0)
    results["single"] = times_single
    if verbose:
        print("Demo 3: zero duration (single time)")
        print("  only time:", times_single[0].iso)
        print("  len:", len(times_single))
        print()

    # Demo 4: using numeric GPS t0
    t0_time = Time("2027-01-01 12:00:00", scale="utc")
    times_gps = get_times(duration=(30, "min"), freq=(5, "min"), t0=t0_time.gps)
    results["gps"] = times_gps
    if verbose:
        print("Demo 4: using numeric GPS t0")
        print("  first (from GPS):", times_gps[0].iso)
        print("  last:", times_gps[-1].iso)
        print("  len:", len(times_gps))
        print()

    # Demo 5: middle time tm, duration exact, freq adjusted if needed
    tm = Time("2027-01-01 12:00:00", scale="utc")
    times_mid = get_times(duration=(2, "hour"), freq=(10, "min"), tm=tm)
    results["midpoint"] = times_mid
    if verbose:
        print("Demo 5: centered on tm (2 hours, ~10 min steps)")
        print("  tm (middle):", tm.iso)
        print("  first:", times_mid[0].iso)
        print("  last:", times_mid[-1].iso)
        print("  number of steps:", len(times_mid))
        mid_idx = len(times_mid) // 2
        print("  middle element:", times_mid[mid_idx].iso)
        print("  dt[0->1]:", (times_mid[1] - times_mid[0]).to("s"))
        print("  dt[end-1->end]:", (times_mid[-1] - times_mid[-2]).to("s"))
        print("  all times:")
        for t in times_mid:
            print("   ", t.iso)
        print()

    return results


def main(verbose=True):
    return demo_get_times(verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)
