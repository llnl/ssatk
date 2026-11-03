"""Inspect and reject an inconsistent scalar measurement with the existing EKF."""

import numpy as np

from ssatk.navigation import CartesianMeasurement, EKFState, ExtendedKalmanFilter

GALLERY_INCLUDE = False


def main():
    filter = ExtendedKalmanFilter(EKFState([0.0], [[4.0]]))
    # A one-dimensional 3-sigma gate corresponds to NIS <= 9.
    filter.update([10.0], CartesianMeasurement((0,)), np.array([[1.0]]), max_nis=9.0)
    print(filter.last_update)
    print("Retained estimate:", filter.state.x)


if __name__ == "__main__":
    main()
