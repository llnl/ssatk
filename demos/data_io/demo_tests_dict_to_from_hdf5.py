#!/usr/bin/env python3

import os
import datetime as dt
import numpy as np

from ssatk.io.dict_to_from_hdf5 import save_dict_to_hdf5, load_dict_from_hdf5

try:
    from astropy.time import Time as AstroTime
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False



# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False
def build_test_dict():
    data = {
        "simple_array": np.arange(10),
        "scalar_int": 42,
        "scalar_float": 3.14159,
        "string": "hello, HDF5",
        "bytes_val": b"\x00\x01\x02",
        "now": dt.datetime.now(),
        "today": dt.date.today(),
        "time_only": dt.time(12, 34, 56, 789),
        # Numeric lists — new behavior: stored and returned as numpy arrays
        "numeric_list": np.array([1, 2, 3, 4, 5]),
        "float_list": np.array([1.1, 2.2, 3.3]),
        # Non-numeric / ragged lists — still stored as groups, returned as lists
        "nested_lists": [[1, 2, 3], [4, 5], [], [[10, 11], [12]]],
        "mixed": {
            # Numeric sublists stored as arrays
            "list_of_arrays": [np.array([0, 1]), np.array([2, 3, 4])],
            # Mixed bool+float: store as array explicitly
            "list_of_scalars": np.array([1.0, 0.0, 1.23]),
        },
    }

    if HAS_ASTROPY:
        data["astro_time_scalar"] = AstroTime("2024-01-01T00:00:00", scale="utc")
        data["astro_time_array"] = AstroTime(["2024-01-01T00:00:00", "2024-01-02T12:00:00"], scale="utc")

    return data


def main():
    filename = "demo_test.h5"

    if os.path.exists(filename):
        os.remove(filename)

    original = build_test_dict()
    save_dict_to_hdf5(filename, original, mode="w")
    loaded = load_dict_from_hdf5(filename)

    # --- Numeric arrays ---
    assert np.array_equal(loaded["simple_array"], original["simple_array"]), "simple_array mismatch"
    assert np.array_equal(loaded["numeric_list"], original["numeric_list"]), "numeric_list mismatch"
    assert np.array_equal(loaded["float_list"], original["float_list"]), "float_list mismatch"

    # --- Scalars ---
    assert loaded["scalar_int"] == original["scalar_int"], "scalar_int mismatch"
    assert np.isclose(loaded["scalar_float"], original["scalar_float"]), "scalar_float mismatch"

    # --- String / bytes ---
    assert loaded["string"] == original["string"], "string mismatch"
    assert loaded["bytes_val"] == original["bytes_val"], "bytes_val mismatch"

    # --- Datetime ---
    assert loaded["today"] == original["today"], "today mismatch"
    assert loaded["now"] == original["now"], "now mismatch"
    assert loaded["time_only"] == original["time_only"], "time_only mismatch"

    # --- Nested ragged list: outer structure is preserved as a list of lists/arrays ---
    assert len(loaded["nested_lists"]) == len(original["nested_lists"]), "nested_lists length mismatch"
    assert np.array_equal(loaded["nested_lists"][0], [1, 2, 3]), "nested_lists[0] mismatch"
    assert np.array_equal(loaded["nested_lists"][1], [4, 5]), "nested_lists[1] mismatch"
    assert len(loaded["nested_lists"][2]) == 0, "nested_lists[2] mismatch"
    assert np.array_equal(loaded["nested_lists"][3][0], [10, 11]), "nested_lists[3][0] mismatch"
    assert np.array_equal(loaded["nested_lists"][3][1], [12]), "nested_lists[3][1] mismatch"

    # --- Mixed nested dict ---
    assert np.array_equal(loaded["mixed"]["list_of_arrays"][0], original["mixed"]["list_of_arrays"][0]), \
        "mixed/list_of_arrays[0] mismatch"
    assert np.array_equal(loaded["mixed"]["list_of_arrays"][1], original["mixed"]["list_of_arrays"][1]), \
        "mixed/list_of_arrays[1] mismatch"
    assert np.array_equal(loaded["mixed"]["list_of_scalars"], original["mixed"]["list_of_scalars"]), \
        "mixed/list_of_scalars mismatch"

    # --- Astropy Time ---
    if HAS_ASTROPY:
        o_scalar = original["astro_time_scalar"]
        l_scalar = loaded["astro_time_scalar"]
        assert np.isclose(o_scalar.mjd, l_scalar.mjd), "astro_time_scalar mjd mismatch"
        assert o_scalar.scale == l_scalar.scale, "astro_time_scalar scale mismatch"

        o_arr = original["astro_time_array"]
        l_arr = loaded["astro_time_array"]
        assert np.allclose(o_arr.mjd, l_arr.mjd), "astro_time_array mjd mismatch"
        assert o_arr.scale == l_arr.scale, "astro_time_array scale mismatch"

    print("All assertions passed.")

    if os.path.exists(filename):
        os.remove(filename)

    return {"original": original, "loaded": loaded}


if __name__ == "__main__":
    main()
