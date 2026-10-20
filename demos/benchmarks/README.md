# Propagation benchmarks

These demos compare SSATK with reproducible reference sources and optional
external propagators. Generated figures belong under
`~/ssatk_output/figures/benchmarks/`.
Benchmark summaries and generated reference state data belong under
`~/ssatk_output/data/benchmarks/`.
When figures are enabled, the demos also generate the captioned PDF report
`~/ssatk_output/documents/benchmarks/ssatk_propagation_benchmark_report.pdf`.

- `demo_artemis_benchmark.py` uses JPL Horizons Artemis II/Orion state vectors
  and reports SSAPy Kepler propagation residuals. With the split data packages installed, it
  matches executed NASA maneuver events at the nearest hourly sample; pass
  `match_burns=False` to retain the legacy position-threshold auto-sync.
- `demo_orekit_benchmark.py` runs Orekit's Java `KeplerianPropagator` and
  compares its Cartesian states with SSATK. It downloads Orekit 10.3.1 through
  Maven only when the demo is run outside pytest and Maven is available.
- `demo_gmat_benchmark.py` runs GMAT R2026a's `RungeKutta89` console propagator
  in Ubuntu 24.04 Podman and compares its point-mass Cartesian states with SSATK.
- `demo_long_term_propagation_benchmark.py` compares matched Earth-centered
  degree/order-0 cases over 7 days at low Earth orbit (LEO), 30 days at
  geostationary orbit (GEO), and 30 days at cislunar radius against GMAT and
  Orekit. The cislunar case is a two-body regional comparison, not an
  Earth-Moon-Sun model.
- `demo_nbody_propagation_benchmark.py` repeats those regimes with Earth–Moon–Sun
  and full planetary point-mass ladders. The benchmark uses DE430 for SSAPy,
  GMAT SPICE, and Orekit. Set `GMAT_DE430_BSP` and `OREKIT_DATA_DIR` when the
  shared `/p/lustre1/yeager7` benchmark data are not available. The supplied
  GMAT DE430 kernel covers the Earth–Moon–Sun ladder; the full planetary ladder
  remains available through Orekit because GMAT also requires separate
  satellite-center kernels for those bodies.

STK and FreeFlyer require separate licensed/native installations and are not
silently substituted by this benchmark suite. GMAT is optional; when its
installation and Podman runtime are unavailable, the GMAT demo skips cleanly.
