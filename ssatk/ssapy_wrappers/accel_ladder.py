def ssapy_accel_ladder(area=1.0, mass=250.0, CR=1.2, CD=2.2):
    """
    Returns a dict: name -> ssapy Accel object.

    Ordering is least -> most complete:
      1) Kepler Earth point mass
      2) + Moon point mass
      3) + Earth J2
      4) + Sun
      5) + Planets
      6) Upgrade Earth gravity: J2 -> full harmonics (keep Moon/Sun/Planets)
      7) + Moon harmonics
      8) + SRP
      9) + EarthRad
    """
    from ssapy.accel import AccelKepler, AccelSolRad, AccelEarthRad
    from ssapy.body import get_body
    from ssapy.gravity import AccelHarmonic, AccelThirdBody

    earth = get_body("Earth", model="EGM2008")
    moon = get_body("moon")
    sun = get_body("Sun")
    planets = [get_body(p) for p in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")]

    a_kepler = AccelKepler()

    a_earth_j2 = AccelHarmonic(earth, 2, 0)
    a_earth_full = AccelHarmonic(earth, 140, 140)

    a_moon_point = AccelThirdBody(moon)
    a_moon_full = AccelHarmonic(moon, 20, 20)

    a_sun = AccelThirdBody(sun)

    a_planets = None
    for p in planets:
        term = AccelThirdBody(p)
        a_planets = term if a_planets is None else a_planets + term

    a_solrad = AccelSolRad(area=float(area), mass=float(mass), CR=float(CR))
    a_earthrad = AccelEarthRad(area=float(area), mass=float(mass), CR=float(CR))

    suite = {}

    # 1) Kepler only
    suite["Kep"] = a_kepler

    # 2) + Moon point mass
    m2 = a_kepler + a_moon_point
    suite["Kep+Moon"] = m2

    # 3) + Earth J2
    m3 = m2 + a_earth_j2
    suite["Kep+Moon+J2"] = m3

    # 4) + Sun
    m4 = m3 + a_sun
    suite["Kep+Moon+J2+Sun"] = m4

    # 5) + Planets (shortened)
    m5 = m4 + a_planets
    suite["Kep+Moon+J2+Sun+Pln"] = m5

    # 6) Upgrade Earth gravity: full harmonics, keep Moon/Sun/Planets (shortened)
    m6 = a_kepler + a_earth_full + a_moon_point + a_sun + a_planets
    suite["EH140+Moon+Sun+Pln"] = m6

    # 7) + Moon harmonics
    m7 = m6 + a_moon_full
    suite["EH140+MH20+Sun+Pln"] = m7

    # 8) + SRP
    m8 = m7 + a_solrad
    suite["EH140+MH20+Sun+Pln+SRP"] = m8

    # 9) + Earth radiation pressure
    m9 = m8 + a_earthrad
    suite["EH140+MH20+Sun+Pln+SRP+ERad"] = m9

    return suite
