"""Synthetic corrected three-way/ramped TDM fit and differenced Doppler."""

from dataclasses import replace

import numpy as np
from astropy.time import Time

from ssatk.constants import EARTH_MU
from ssatk.coordinates.geodetic import llh_to_gcrf
from ssatk.io import read_tdm
from ssatk.navigation import GroundStation, GroundStationSensor, RadioTrackingConfig, solve_batch_orbit
from ssatk.plots.plotutils import figsave
from ssatk.propagators_orbit import propagate_orbit_state_with_stm

GALLERY_CATEGORY = "navigation"


def main(make_figures=True):
    t0 = 1.4e9
    times = t0 + np.arange(30., 151., 20.)
    # Weather, TEC and hardware values are prescribed synthetic inputs, not flight calibration.
    transmitter = GroundStation(-104.9, 39.7, elevation_m=1600, min_elevation_rad=np.deg2rad(10),
        clock_offset_s=1e-6, clock_drift=1e-12, clock_epoch=t0, transmit_delay_s=3e-6,
        meteorology=(850., 285., 5.), slant_tec=10e16)
    receivers = {
        "WEST": GroundStation(-110., 39., elevation_m=1600, min_elevation_rad=np.deg2rad(10),
            clock_offset_s=-.5e-6, clock_epoch=t0, receive_delay_s=2e-6, meteorology=(850., 285., 5.), slant_tec=12e16),
        "EAST": GroundStation(-100., 35., elevation_m=1200, min_elevation_rad=np.deg2rad(10),
            meteorology=(900., 290., 4.), slant_tec=8e16),
    }
    station_position, _ = llh_to_gcrf(transmitter.lon_deg, transmitter.lat_deg, t0, height=transmitter.elevation_m)
    radial = np.asarray(station_position) / np.linalg.norm(station_position)
    east = np.cross([0., 0., 1.], radial)
    east /= np.linalg.norm(east)
    truth = np.r_[7e6 * radial, np.sqrt(EARTH_MU / 7e6) * east]
    trajectory = propagate_orbit_state_with_stm(r0=truth[:3], v0=truth[3:], t0=t0,
        times=np.r_[t0, times, times[-1]+1], dense_output=True, rtol=1e-12, atol=1e-9)
    hardware = RadioTrackingConfig(7.2e9, 10., 880/749, 2e-6, transmitter=transmitter,
                                   transponder_delay_is_proper=True, shapiro_bodies=("earth", "sun"))
    radio = replace(hardware, frequency_ramps=[
        [t0-60, t0+85, 7.2e9, 40.], [t0+85, t0+240, 7.2e9+5000, -20.],
    ])
    def epoch(time):
        value = Time(time, format="gps")
        value.precision = 9
        return value.utc.isot
    lines = ["CCSDS_TDM_VERS = 2.0", f"CREATION_DATE = {epoch(t0)}", "ORIGINATOR = SSATK_SYNTHETIC_DEMO"]
    values = {}
    for name, station in receivers.items():
        lines.extend(["META_START", "TIME_SYSTEM = UTC", f"STOP_TIME = {epoch(t0+240)}",
            "PARTICIPANT_1 = TX", "PARTICIPANT_2 = SPACECRAFT", f"PARTICIPANT_3 = {name}",
            "MODE = SEQUENTIAL", "PATH = 1,2,3", "RANGE_UNITS = km", "INTEGRATION_INTERVAL = 10",
            "INTEGRATION_REF = END", f"FREQ_OFFSET = {radio.reference_frequency}", "META_STOP", "DATA_START"])
        for start, _, frequency, rate in radio.frequency_ramps:
            lines.extend([f"TRANSMIT_FREQ_1 = {epoch(start)} {frequency:.17g}",
                          f"TRANSMIT_FREQ_RATE_1 = {epoch(start)} {rate:.17g}"])
        for kind, keyword in (("three_way_range", "RANGE"), ("three_way_doppler", "RECEIVE_FREQ_1")):
            sensor = GroundStationSensor(station, 0., measurement=kind, radio=radio, rng=0)
            values[name, kind] = []
            for time in times:
                observation = sensor.measure(trajectory, time)
                if not observation.valid:
                    raise RuntimeError("synthetic corrected link is outside its elevation/visibility bounds.")
                raw = observation.value / 500 if keyword == "RANGE" else observation.value
                lines.append(f"{keyword} = {epoch(time)} {raw:.17g}")
                values[name, kind].append(observation.value)
        lines.append("DATA_STOP")
    stations = {"TX": transmitter, **receivers}
    # Ramps are reconstructed from the TDM rather than supplied a second time.
    observations = read_tdm('\n'.join(lines)).to_station_observations(stations,
        radio={"TX": replace(hardware, transmitter=None)},
        covariance={"three_way_range": 1., "three_way_doppler": .0001}, range_convention="round_trip")
    fit = solve_batch_orbit(observations, truth+[100., -100., 100., .1, -.1, .1], t0=t0,
                            propagation_kwargs={"rtol": 1e-12, "atol": 1e-9}, max_nfev=30)
    difference_radio = replace(radio, reference_receiver=receivers["EAST"], reference_radio=radio)
    differences = np.array([receivers["WEST"].predict_radio(trajectory, time, "differenced_doppler", difference_radio).value for time in times])
    error = fit.state0 - truth
    print(f"Synthetic corrected three-way TDM fit: success={fit.success}, rank={fit.numerical_rank}, "
          f"position_error={np.linalg.norm(error[:3]):.6g} m, velocity_error={np.linalg.norm(error[3:]):.6g} m/s")
    if not fit.success:
        raise RuntimeError(fit.message)
    if make_figures:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        for name in receivers:
            axes[0].plot(times-t0, values[name, "three_way_doppler"], label=name)
        axes[0].set(ylabel="Three-way Doppler (Hz)", title="Synthetic ramps, media, clocks and relativistic corrections")
        axes[0].legend()
        axes[1].plot(times-t0, differences)
        axes[1].set(xlabel="Receive clock epoch since t0 (s)", ylabel="WEST minus EAST (Hz)")
        for axis in axes:
            axis.grid(alpha=.3)
        figure.tight_layout()
        # gallery-call: demo_advanced_radio_tracking.png
        figsave(figure, "demo_advanced_radio_tracking.png")
        # gallery-end
        plt.close(figure)
    return {"fit": fit, "truth": truth, "state_error": error, "differenced_doppler_hz": differences}


if __name__ == "__main__":
    main()
