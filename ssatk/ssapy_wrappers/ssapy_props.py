from ssapy.accel import AccelKepler, AccelSolRad, AccelEarthRad, AccelDrag
from ssapy.body import get_body
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator, KeplerianPropagator


def ssapy_kwargs(mass=250.0, area=0.022, CD=2.3, CR=1.3):
    """Default space-object parameters used by SSAPy non-conservative models."""
    return {"mass": float(mass), "area": float(area), "CD": float(CD), "CR": float(CR)}


def keplerian_prop(ode_kwargs=None):
    return KeplerianPropagator()


def keplerian_numerical_prop():
    return SciPyPropagator(AccelKepler())


_accel_3_cache = None


def threebody_prop(ode_kwargs=None):
    global _accel_3_cache
    if _accel_3_cache is None:
        _accel_3_cache = AccelKepler() + AccelThirdBody(get_body("moon"))
    return SciPyPropagator(_accel_3_cache, ode_kwargs=ode_kwargs)


_accel_4_cache = None


def fourbody_prop(ode_kwargs=None):
    global _accel_4_cache
    if _accel_4_cache is None:
        _accel_4_cache = (
            AccelKepler()
            + AccelThirdBody(get_body("moon"))
            + AccelThirdBody(get_body("Sun"))
        )
    return SciPyPropagator(_accel_4_cache, ode_kwargs=ode_kwargs)


_accel_best_cache = None


def best_prop(kwargs=None, ode_kwargs=None):
    """
    "Best-like" force model: Earth(Kepler+EGM2008 140x140) + Moon(point+20x20) + Sun + planets + SRP+EarthRad+Drag
    """
    global _accel_best_cache

    if kwargs is None:
        kwargs = ssapy_kwargs()

    if _accel_best_cache is None:
        aEarth = AccelKepler() + AccelHarmonic(get_body("Earth", model="EGM2008"), 140, 140)

        moon = get_body("moon")
        aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)

        aSun = AccelThirdBody(get_body("Sun"))

        planet_names = ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
        planets = None
        for name in planet_names:
            term = AccelThirdBody(get_body(name))
            planets = term if planets is None else planets + term

        nonConservative = AccelSolRad(**kwargs) + AccelEarthRad(**kwargs) + AccelDrag(**kwargs)

        _accel_best_cache = aEarth + aMoon + aSun + planets + nonConservative

    return SciPyPropagator(_accel_best_cache, ode_kwargs=ode_kwargs)


_accel_best_gravity_cache = None


def best_gravity_prop(kwargs=None, ode_kwargs=None):
    """
    "Best-like" force model: Earth(Kepler+EGM2008 140x140) + Moon(point+20x20) + Sun + planets + SRP+EarthRad+Drag
    """
    global _accel_best_gravity_cache

    if kwargs is None:
        kwargs = ssapy_kwargs()

    if _accel_best_gravity_cache is None:
        aEarth = AccelKepler() + AccelHarmonic(get_body("Earth", model="EGM2008"), 140, 140)

        moon = get_body("moon")
        aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)

        aSun = AccelThirdBody(get_body("Sun"))

        planet_names = ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
        planets = None
        for name in planet_names:
            term = AccelThirdBody(get_body(name))
            planets = term if planets is None else planets + term

        nonConservative = AccelSolRad(**kwargs) + AccelEarthRad(**kwargs)

        _accel_best_gravity_cache = aEarth + aMoon + aSun + planets + nonConservative

    return SciPyPropagator(_accel_best_gravity_cache, ode_kwargs=ode_kwargs)


def ssapy_prop(propkw=None, ode_kwargs=None):
    """
    A lighter "ssapy_prop"-style model (no planets, no drag): Earth full + Moon full + Sun + SRP + EarthRad
    """
    if propkw is None:
        propkw = ssapy_kwargs()

    moon = get_body("moon")
    sun = get_body("Sun")
    earth = get_body("Earth", model="EGM2008")

    aEarth = AccelKepler() + AccelHarmonic(earth, 140, 140)
    aSun = AccelThirdBody(sun)
    aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
    aSolRad = AccelSolRad(**propkw)
    aEarthRad = AccelEarthRad(**propkw)

    accel = aEarth + aMoon + aSun + aSolRad + aEarthRad
    return SciPyPropagator(accel, ode_kwargs=ode_kwargs)


if __name__ == "__main__":
    # quick smoke test: build each propagator (doesn't propagate anything)
    props = {
        "kepler": keplerian_prop(),
        "3body": threebody_prop(),
        "4body": fourbody_prop(),
        "ssapy_prop": ssapy_prop(),
        "best": best_prop(),
    }
    for k, p in props.items():
        print(k, "->", p)
