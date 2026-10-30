"""Compute collision probability from a CDM and an explicit combined radius.

Assumes independent Gaussian object errors, a circular hard-body cross-section
and a short rectilinear encounter. The CDM does not establish covariance realism
or suitability for slow encounters.
"""

import argparse

from ssatk.io.ccsds_cdm import read_cdm
from ssatk.ssa import encounter_frame, probability_of_collision, relative_encounter_covariance

GALLERY_INCLUDE = False


def calculate(source, combined_radius_m):
    message = read_cdm(source)
    first, second = message.object1, message.object2
    relative = second.state_gcrf() - first.state_gcrf()
    basis = encounter_frame(relative[:3], relative[3:])
    covariance = relative_encounter_covariance(
        first.position_covariance_gcrf(), second.position_covariance_gcrf(), basis,
    )
    return probability_of_collision(basis.T @ relative[:3], covariance, combined_radius_m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cdm", help="CCSDS CDM file")
    parser.add_argument("--combined-radius-m", type=float, required=True,
                        help="Sum of the two physical object radii in metres")
    args = parser.parse_args()
    print(f"Collision probability: {calculate(args.cdm, args.combined_radius_m):.10g}")


if __name__ == "__main__":
    main()
