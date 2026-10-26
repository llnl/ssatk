"""Catalog of physics capabilities used by the interactive gallery explorer."""

TOPICS = [
    {
        "id": "shadows",
        "title": "Shadows and eclipses",
        "kicker": "Finite-disk occultation geometry",
        "summary": "Evaluate cylindrical or conical Earth/Moon shadowing and solar-disk visibility along an ephemeris.",
        "chain": ["spacecraft and Sun positions", "angular body disks", "overlap or shadow test", "illumination fraction"],
        "apis": [
            "ssatk.environment.cylindrical_eclipse_fraction",
            "ssatk.environment.solar_occultation_fraction",
            "ssatk.environment.solar_disk_visible_fraction",
            "ssatk.environment.SpaceEnvironment",
            "ssatk.plots.coverage_analysis.compute_eclipse",
        ],
        "limits": "The cylindrical helper is binary. Conical occultation uses spherical bodies and finite Sun/Earth/Moon disks; it does not model terrain, rings, spacecraft self-shadowing, or atmosphere.",
        "demos": ["eclipse/demo_eclipse_views", "orbit_visualization/demo_groundtrack_enhanced", "photometry/demo_lambertian_brightness"],
    },
    {
        "id": "lighting",
        "title": "Lighting and photometry",
        "kicker": "Reflected, thermal, and facet brightness",
        "summary": "Compare Lambertian sphere brightness with attitude-dependent faceted light curves across optical and infrared bands.",
        "chain": ["Sun/Earth/Moon geometry", "phase and visibility", "surface reflectance", "band irradiance and AB magnitude"],
        "apis": [
            "ssatk.compute.lambertian_magnitude.lambertian_reflection",
            "ssatk.compute.lambertian_magnitude.lambertian_sphere_brightness",
            "ssatk.compute.faceted_magnitude.faceted_reflection",
            "ssatk.compute.faceted_magnitude.faceted_light_curve",
            "ssatk.compute.faceted_magnitude.facet_scattering_area",
        ],
        "limits": "The sphere model is Lambertian and the faceted model uses diffuse reflection; specular glints, clouds, terrain, and calibrated detector response are outside these APIs.",
        "demos": ["photometry/demo_lambertian_brightness", "photometry/demo_faceted_light_curves", "photometry/demo_photometry_application"],
    },
    {
        "id": "thermal",
        "title": "Thermal emission and recoil",
        "kicker": "Prescribed temperature and optical momentum transfer",
        "summary": "Compute gray-body emission at a supplied temperature. Separately, optical reemission coefficients describe thermal recoil in solar-radiation-pressure (SRP) force and torque models; they do not use the emission calculation's temperature.",
        "chain": ["prescribed temperature", "Planck band fraction", "thermal irradiance"],
        "apis": [
            "ssatk.compute.lambertian_magnitude.thermal_emission",
            "ssatk.compute.lambertian_magnitude.lambertian_sphere_brightness",
            "ssatk.accelerations_6dof.spacecraft.flat_plate_srp_acceleration_torque",
            "ssatk.accelerations_6dof.spacecraft.facet_srp_acceleration_torque",
            "ssatk.accelerations_6dof.spacecraft.SpacecraftFlatPlateSolRad",
        ],
        "limits": "Temperature is an input. SSATK has no spacecraft temperature state, heat-conduction solver, thermal inertia model, or radiative thermal-balance evolution. thermal_reemission is an SRP momentum coefficient, not a temperature solver.",
        "demos": ["photometry/demo_lambertian_brightness", "six_dof/demo_6dof_srp_facets"],
    },
    {
        "id": "attitude",
        "title": "Attitude and force coupling",
        "kicker": "Six degrees of freedom (6-DoF)",
        "summary": "Propagate translation and rigid-body attitude together. Gravity-gradient torque, solar-radiation pressure on plates or facets, and attitude control connect orientation to motion, with optional mass and reaction-wheel states.",
        "chain": ["body geometry and inertia", "quaternion attitude", "force and torque models", "coupled 6-DoF trajectory"],
        "apis": [
            "ssatk.propagators_6dof.sixdof.Spacecraft",
            "ssatk.propagators_6dof.sixdof.propagate_6dof",
            "ssatk.accelerations_6dof.spacecraft.SpacecraftAttitudePD",
            "ssatk.satellites.SpacecraftBody",
            "ssatk.satellites.Facet",
        ],
        "limits": "The built-in attitude controller is a basic body-frame quaternion PD controller, not a complete flight-software or guidance stack. Flexible-body and slosh dynamics require the separate extended propagator APIs.",
        "demos": ["six_dof/demo_6dof_srp_facets", "six_dof/demo_spacecraft_view"],
    },
]

__all__ = ["TOPICS"]
