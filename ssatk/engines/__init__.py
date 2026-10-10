"""Engine and spacecraft propulsion catalogs.

This package is the canonical home for satellite propulsion presets, thrust
profiles, propellant estimates, and stationkeeping/maneuver engine helpers.
"""

from ssatk._namespace import import_public_modules

from .catalog import thruster_catalog_dict, thruster_spec

thruster_specs = thruster_catalog_dict()

import_public_modules(__name__, __file__, globals())
