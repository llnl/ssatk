from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ssatk.compute import lambertian_magnitude as photometry


class PhotometryVisibilityTests(unittest.TestCase):
    def geometry(self, object_position, observer):
        with patch('ssapy.utils.sunPos', return_value=np.array([0., 1.5e11, 0.])), \
             patch('ssapy.utils.moonPos', return_value=np.array([0., 3.8e8, 0.])):
            return photometry._setup(object_position, observer, photometry.DEFAULT_TIME, 'V', 0.16,
                                      None, None, 0., photometry.R_EARTH, 100e3)

    def package(self, geometry):
        return photometry._package(geometry, {'sun': 1.}, {'sun': 0.1}, {},
                                    SimpleNamespace(isot='synthetic geometry'), photometry.F_NU_AB_ZERO)

    def test_space_observer_through_earth_has_zero_observed_flux(self):
        geometry = self.geometry([-7e6, 0, 0], [7e6, 0, 0])
        result = self.package(geometry)
        self.assertTrue(result['object_below_horizon'])
        self.assertEqual(result['irradiance_inband_total_at_observer_W_m2'], 0.)
        self.assertEqual(result['ab_mag_observed'], np.inf)
        self.assertTrue(np.isfinite(result['ab_mag_exoatmospheric']))

    def test_ground_below_horizon_flag_masks_flux_even_without_extinction(self):
        geometry = self.geometry([8e6, 0, 0], [7e6, 0, 0])
        geometry['below_horizon'] = True
        geometry['extinction_mag'] = 0.
        result = self.package(geometry)
        self.assertEqual(result['irradiance_inband_total_at_observer_W_m2'], 0.)
        self.assertEqual(result['ab_mag_observed'], np.inf)
