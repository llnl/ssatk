"""Reference launch vehicle stage data."""

launch_vehicles = {
    "Falcon 9 Full Thrust": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "Merlin 1D",
                        "count": 9,
                        "ISP_SL": 282,  # seconds
                        "ISP_vac": 311,  # seconds
                        "thrust_SL": 845,  # kN per engine
                        "thrust_vac": 981  # kN per engine
                    }
                ],
                "mass_empty": 25600,  # kg
                "mass_propellant": 395700  # kg
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "Merlin 1D Vacuum",
                        "count": 1,
                        "ISP_vac": 348,  # seconds
                        "thrust_vac": 934  # kN
                    }
                ],
                "mass_empty": 3900,  # kg
                "mass_propellant": 92670  # kg
            }
        ],
        "total_mass": 549000  # kg
    },
    "Atlas V": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "RD-180",
                        "count": 1,
                        "ISP_SL": 311.3,  # seconds
                        "ISP_vac": 337.8,  # seconds
                        "thrust_SL": 3827,  # kN
                        "thrust_vac": 4152  # kN
                    }
                ],
                "mass_empty": 21054,  # kg
                "mass_propellant": 284089  # kg
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "RL10A",
                        "count": 1,
                        "ISP_vac": 450.5,  # seconds
                        "thrust_vac": 99.2  # kN
                    }
                ],
                "mass_empty": 2316,  # kg
                "mass_propellant": 20830  # kg
            }
        ]
    },
    "Soyuz-2.1b": {
        "stages": [
            {
                "stage_number": 1,
                "engines": [
                    {
                        "name": "RD-107A",
                        "count": 4,
                        "ISP_SL": 262,  # seconds
                        "ISP_vac": 319,  # seconds
                        "thrust_SL": 838.5,  # kN per engine
                        "thrust_vac": 1021.3  # kN per engine
                    }
                ],
                "mass_empty": 15136,  # kg (total for 4 boosters)
                "mass_propellant": 156640  # kg (total for 4 boosters)
            },
            {
                "stage_number": 2,
                "engines": [
                    {
                        "name": "RD-108A",
                        "count": 1,
                        "ISP_SL": 255,  # seconds
                        "ISP_vac": 319,  # seconds
                        "thrust_SL": 792.5,  # kN
                        "thrust_vac": 990.2  # kN
                    }
                ],
                "mass_empty": 6545,  # kg
                "mass_propellant": 90100  # kg
            },
            {
                "stage_number": 3,
                "engines": [
                    {
                        "name": "RD-0124",
                        "count": 1,
                        "ISP_vac": 359,  # seconds
                        "thrust_vac": 297.9  # kN
                    }
                ],
                "mass_empty": 2355,  # kg
                "mass_propellant": 25400  # kg
            }
        ]
    }
}

def assemble_stage_stack(payload, stages=(), *, structures=()):
    """Assemble existing SpacecraftBody stages around a payload body.

    All inputs use the same body axes and origin, metres and kg; each dry
    body's inertia is about its own center_of_mass. Tank names must be unique
    and nonempty so inventory remains unambiguous. Omit detached stages or
    structures and rebuild to obtain the retained body's mass properties.
    This is configuration accounting, not a separation event/state reset.
    Catalog masses alone do not supply geometry or calibrated inertia.
    The payload's ``reference_area`` is retained unchanged; no combined
    aerodynamic reference area is inferred for the assembled stack.
    """
    from ssatk.satellites import Component, SpacecraftBody

    stages = tuple(stages)
    bodies = (payload,) + stages
    if any(not isinstance(body, SpacecraftBody) for body in bodies):
        raise TypeError("payload and stages must be SpacecraftBody objects.")
    structures = tuple(structures)
    if any(not isinstance(component, Component) for component in structures):
        raise TypeError("structures must be Component objects.")
    names = [tank.name for body in bodies for tank in body.tanks]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("stage-stack tank names must be unique and nonempty.")
    return payload.with_updates(
        components=payload.components + structures + tuple(
            Component(name=stage.name, mass=stage.mass,
                      position_body=stage.center_of_mass, inertia=stage.inertia)
            for stage in stages
        ) + tuple(component for stage in stages for component in stage.components),
        **{field: tuple(item for body in bodies for item in getattr(body, field))
           for field in ("tanks", "facets", "thrusters", "magnetic_dipoles", "reaction_wheels")},
    )


__all__ = ["launch_vehicles", "assemble_stage_stack"]
