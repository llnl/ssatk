"""Compare daily and storm-time MSIS modes using the same packaged history."""

from astropy.time import Time

from ssatk.environment import SpaceEnvironment

GALLERY_INCLUDE = False


def main():
    epoch = Time("2024-01-15T12:00:00", scale="utc")
    for mode in ("daily", "storm"):
        environment = SpaceEnvironment(nrlmsise_geomagnetic_mode=mode)
        print(environment.check_data_coverage([epoch.gps]))
        density = environment.nrlmsise00_density(epoch.gps, [6_778_137.0, 0.0, 0.0])
        print(f"{mode}: {density:.8g} kg/m^3")


if __name__ == "__main__":
    main()
