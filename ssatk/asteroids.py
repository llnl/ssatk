import numpy as np

def radius_from_H_albedo(H: np.ndarray, albedo: float = 0.1) -> np.ndarray:
    """
    Calculate asteroid radius from H magnitude and albedo.
    """
    # http://www.physics.sfasu.edu/astro/asteroids/sizemagnitude.html
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10 ** (-0.2 * H)
    return radius


def H_mag(radius: np.ndarray, albedo: float) -> np.ndarray:
    """
    Calculate the H magnitude from the asteroid radius and albedo.
    """
    return 5 * np.log(664500 / (radius * np.sqrt(albedo))) / np.log(10)


def johnsonV_to_lsst_array(M_app: np.ndarray, filters, ast_types: np.ndarray) -> np.ndarray:
    """
    Convert apparent magnitude in Johnson V to LSST filters.

    Parameters
    ----------
    filters : list
        List of filter names ('u', 'g', 'r', 'i', 'z', 'y').
    ast_types : np.ndarray
        Array of asteroid types (0 or 1).
    """
    corrections = np.zeros(np.shape(M_app))
    f = np.array(filters)
    at = np.array(ast_types)

    corrections[np.where((f == 'u') & (at == 0))] = -1.614
    corrections[np.where((f == 'u') & (at == 1))] = -1.927
    corrections[np.where((f == 'g') & (at == 0))] = -0.302
    corrections[np.where((f == 'g') & (at == 1))] = -0.395
    corrections[np.where((f == 'r') & (at == 0))] = 0.172
    corrections[np.where((f == 'r') & (at == 1))] = 0.255
    corrections[np.where((f == 'i') & (at == 0))] = 0.291
    corrections[np.where((f == 'i') & (at == 1))] = 0.455
    corrections[np.where((f == 'z') & (at == 0))] = 0.298
    corrections[np.where((f == 'z') & (at == 1))] = 0.401
    corrections[np.where((f == 'y') & (at == 0))] = 0.303
    corrections[np.where((f == 'y') & (at == 1))] = 0.406
    return M_app - corrections


def johnsonV_to_ztf_array(M_app: np.ndarray, filters, ast_types: np.ndarray) -> np.ndarray:
    """
    Convert apparent magnitude in Johnson V to ZTF filters.

    Parameters
    ----------
    filters : list
        List of filter indices (1: 'g', 2: 'r', 3: 'i').
    ast_types : np.ndarray
        Array of asteroid types (0 or 1).
    """
    corrections = np.zeros(np.shape(M_app))
    f = np.array(filters)
    at = np.array(ast_types)

    corrections[np.where((f == 1) & (at == 0))] = -0.302
    corrections[np.where((f == 1) & (at == 1))] = -0.395
    corrections[np.where((f == 2) & (at == 0))] = 0.172
    corrections[np.where((f == 2) & (at == 1))] = 0.255
    corrections[np.where((f == 3) & (at == 0))] = 0.291
    corrections[np.where((f == 3) & (at == 1))] = 0.455
    return M_app - corrections


def get_albedo_array(num: int = 1) -> tuple:
    """
    Generate random albedo values and asteroid types (0 or 1).
    Returns (albedo_array, type_array) as NumPy arrays of length `num`.
    """
    num = int(num)
    albedo_out = np.empty(0, dtype=float)
    ast_type_out = np.empty(0, dtype=int)

    while albedo_out.size < num:
        fd = 0.253
        d = 0.030
        b = 0.168

        albedo = np.random.uniform(0, 1, size=num)
        sample_ys = np.random.uniform(0, 6, size=num)

        c_pdf = fd * (albedo * np.exp(-albedo**2 / (2 * d**2)) / d**2)
        s_pdf = (1 - fd) * (albedo * np.exp(-albedo**2 / (2 * b**2)) / b**2)

        c_mask = sample_ys < c_pdf
        s_mask = sample_ys < s_pdf

        c_albedo = albedo[c_mask]
        s_albedo = albedo[s_mask]

        c_type = np.zeros(c_albedo.size, dtype=int)
        s_type = np.ones(s_albedo.size, dtype=int)

        albedo_batch = np.concatenate((c_albedo, s_albedo))
        type_batch = np.concatenate((c_type, s_type))

        if albedo_out.size == 0:
            albedo_out = albedo_batch
            ast_type_out = type_batch
        else:
            albedo_out = np.hstack((albedo_out, albedo_batch))
            ast_type_out = np.hstack((ast_type_out, type_batch))

    # Shuffle pairs together, then trim to `num`
    idx = np.random.permutation(albedo_out.size)
    albedo_out = albedo_out[idx][:num]
    ast_type_out = ast_type_out[idx][:num]

    return albedo_out, ast_type_out


def granvik_low_slope(x: np.ndarray) -> np.ndarray:
    """Low-slope function for Granvik distribution."""
    return 0.3034 * x - 3.491


def granvik_high_slope(x: np.ndarray) -> np.ndarray:
    """High-slope function for Granvik distribution."""
    return 0.7235 * x - 13.12


def get_neo_H_mag_array(num: int = 1, upper_mag: float = 28, min_mag: float = 10) -> np.ndarray:
    """
    Generate a random array of H magnitudes for Near-Earth Objects (NEOs).
    """
    num = int(num)
    H_mag_out = []
    while np.size(H_mag_out) != num:
        xs = np.random.uniform(min_mag, upper_mag, size=num)
        ys = np.random.uniform(1, 10**granvik_high_slope(upper_mag), size=num)
        xs_low = xs[np.where(xs < 23)]
        ys_low = ys[np.where(xs < 23)]
        xs_high = xs[np.where(xs >= 23)]
        ys_high = ys[np.where(xs >= 23)]
        index_low = np.where(ys_low < 10**granvik_low_slope(xs_low))
        index_high = np.where(ys_high < 10**granvik_high_slope(xs_high))
        H_mag = np.hstack((xs_low[index_low], xs_high[index_high]))
        if np.size(H_mag_out) == 0:
            H_mag_out = H_mag
            continue
        H_mag_out = np.hstack((H_mag_out, H_mag))
        if np.size(H_mag_out) > num:
            H_mag_out = H_mag_out[:num]
    np.random.shuffle(H_mag_out)
    return H_mag_out


def get_eta_radius_albedo_H_array(num: int = 1, upper_mag: float = 28, min_mag: float = 10) -> dict:
    """
    Generate radius, albedo, asteroid type, and H magnitude arrays for ETA dataset.

    Returns
    -------
    dict
        {'radius': ..., 'albedo': ..., 'type': ..., 'H': ...}
    """
    albedo, ast_type = get_albedo_array(num=num)
    H = get_neo_H_mag_array(num=num, upper_mag=upper_mag, min_mag=min_mag)
    radius = 1329e3 / (2 * np.sqrt(albedo)) * 10**(-0.2 * H)
    return {'radius': radius, 'albedo': albedo, 'type': ast_type, 'H': H}
