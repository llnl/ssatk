"""Vector utilities and small helpers for geometry and plotting."""

# flake8: noqa: E501
import matplotlib.pyplot as plt
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def extend_vector(vector: np.ndarray, distance: float) -> np.ndarray:
    """Extend a vector by a given distance along its direction."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot extend a zero vector.")
    unit_vector = vector / norm
    return vector + unit_vector * distance


def getAngle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate the angle between vectors a, b, and c, where b is the vertex.

    Parameters:
    a, b, c (np.ndarray): Input vectors.

    Returns:
    np.ndarray: The angle between the vectors in radians.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    cosine_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.arccos(cosine_angle)


def angle_between_vectors(vector1, vector2):
    """Return the angle between two vectors in radians."""
    return np.arccos(np.clip(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    if vec1_norm == 0 or vec2_norm == 0:
        raise ValueError("Cannot align zero-length vectors.")

    a, b = (vec1 / vec1_norm).reshape(3), (vec2 / vec2_norm).reshape(3)
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)

    if np.isclose(s, 0.0):
        if c > 0:
            return np.eye(3)
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def normed(arr):
    """Normalise an array of vectors along the last axis."""
    return arr / np.sqrt(np.einsum("...i,...i", arr, arr))[..., None]


def einsum_norm(a, indices='ij,ji->i'):
    return np.sqrt(np.einsum(indices, a, a))


def normSq(arr):
    return np.einsum("...i,...i", arr, arr)


def norm(arr):
    return np.sqrt(np.einsum("...i,...i", arr, arr))


def rotate_vector(v_unit, theta, phi, save_path=False, **save_kwargs):
    v_unit = v_unit / np.linalg.norm(v_unit, axis=-1)
    if np.all(np.abs(v_unit) != np.max(np.abs(v_unit))):
        perp_vector = np.cross(v_unit, np.array([1, 0, 0]))
    else:
        perp_vector = np.cross(v_unit, np.array([0, 1, 0]))
    perp_vector /= np.linalg.norm(perp_vector)

    theta = np.radians(theta)
    phi = np.radians(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    R1 = np.array([
        [cos_theta + (1 - cos_theta) * perp_vector[0]**2, 
         (1 - cos_theta) * perp_vector[0] * perp_vector[1] - sin_theta * perp_vector[2], 
         (1 - cos_theta) * perp_vector[0] * perp_vector[2] + sin_theta * perp_vector[1]],
        [(1 - cos_theta) * perp_vector[1] * perp_vector[0] + sin_theta * perp_vector[2], 
         cos_theta + (1 - cos_theta) * perp_vector[1]**2, 
         (1 - cos_theta) * perp_vector[1] * perp_vector[2] - sin_theta * perp_vector[0]],
        [(1 - cos_theta) * perp_vector[2] * perp_vector[0] - sin_theta * perp_vector[1], 
         (1 - cos_theta) * perp_vector[2] * perp_vector[1] + sin_theta * perp_vector[0], 
         cos_theta + (1 - cos_theta) * perp_vector[2]**2]
    ])

    # Apply the rotation matrix to v_unit to get the rotated unit vector
    v1 = np.dot(R1, v_unit)

    # Rotation matrix for rotation about v_unit
    R2 = np.array([[cos_phi + (1 - cos_phi) * v_unit[0]**2,
                    (1 - cos_phi) * v_unit[0] * v_unit[1] - sin_phi * v_unit[2],
                    (1 - cos_phi) * v_unit[0] * v_unit[2] + sin_phi * v_unit[1]],
                   [(1 - cos_phi) * v_unit[1] * v_unit[0] + sin_phi * v_unit[2],
                    cos_phi + (1 - cos_phi) * v_unit[1]**2,
                    (1 - cos_phi) * v_unit[1] * v_unit[2] - sin_phi * v_unit[0]],
                   [(1 - cos_phi) * v_unit[2] * v_unit[0] - sin_phi * v_unit[1],
                    (1 - cos_phi) * v_unit[2] * v_unit[1] + sin_phi * v_unit[0],
                    cos_phi + (1 - cos_phi) * v_unit[2]**2]])

    v2 = np.dot(R2, v1)

    from .plots.plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs

    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "rotate_vector")

    if save_path:
        plt.rcParams.update({'font.size': 9, 'figure.facecolor': 'black'})
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, v_unit[0], v_unit[1], v_unit[2], color='b')
        ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='g')
        ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='r')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.set_facecolor('black')  # Set plot background color to black
        ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
        ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
        ax.tick_params(axis='z', colors='white')  # Set z-axis tick color to white
        ax.set_title('Vector Plot', color='white')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.grid(True)
        from .plots import figsave
        ax.set_title(f'Vector Plot\ntheta: {np.degrees(theta):.0f}, phi: {np.degrees(phi):.0f}', color='white')
        figsave(fig, save_path=save_path)
    return v2 / np.linalg.norm(v2, axis=-1)


def rotate_points_3d(points, axis=None, theta=-np.pi / 2):
    """
    Rotate a set of 3D points about a 3D axis by an angle theta in radians.

    Args:
        points (np.ndarray): The set of 3D points to rotate, as an Nx3 array.
        axis (np.ndarray): The 3D axis to rotate about, as a length-3 array. Default is the z-axis.
        theta (float): The angle to rotate by, in radians. Default is pi/2.

    Returns:
        np.ndarray: The rotated set of 3D points, as an Nx3 array.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.asarray(axis, dtype=float)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")

    # Normalize the axis to be a unit vector
    axis = axis / axis_norm

    # Compute the quaternion representing the rotation
    qw = np.cos(theta / 2)
    qx, qy, qz = axis * np.sin(theta / 2)

    # Construct the rotation matrix from the quaternion
    R = np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    # Apply the rotation matrix to the set of points
    rotated_points = np.dot(R, points.T).T

    return rotated_points


def perpendicular_vectors(v):
    """Returns two vectors that are perpendicular to v and each other."""
    # Check if v is the zero vector
    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Input vector cannot be the zero vector.")

    # Choose an arbitrary non-zero vector w that is not parallel to v
    w = np.array([1., 0., 0.])
    if np.allclose(v, w) or np.allclose(v, -w):
        w = np.array([0., 1., 0.])
    u = np.cross(v, w)
    if np.allclose(u, np.zeros_like(u)):
        w = np.array([0., 0., 1.])
        u = np.cross(v, w)
    w = np.cross(v, u)

    return u, w


def points_on_circle(r, v, rad, num_points=4):
    # Convert inputs to numpy arrays
    r = np.array(r)
    v = np.array(v)

    # Find the perpendicular vectors to the given vector v
    if np.all(v[:2] == 0):
        if np.all(v[2] == 0):
            raise ValueError("The given vector v must not be the zero vector.")
        else:
            u = np.array([1, 0, 0])
    else:
        u = np.array([-v[1], v[0], 0])
    u = u / np.linalg.norm(u)
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-15:
        # v is parallel to z-axis
        w = np.array([0, 1, 0])
    else:
        w = w / w_norm
    # Generate a sequence of angles for equally spaced points
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Compute the x, y, z coordinates of each point on the circle
    x = rad * np.cos(angles) * u[0] + rad * np.sin(angles) * w[0]
    y = rad * np.cos(angles) * u[1] + rad * np.sin(angles) * w[1]
    z = rad * np.cos(angles) * u[2] + rad * np.sin(angles) * w[2]

    # Apply rotation about z-axis by 90 degrees
    rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotated_points = np.dot(rot_matrix, np.column_stack((x, y, z)).T).T

    # Translate the rotated points to the center point r
    points_rotated = rotated_points + r.reshape(1, 3)

    return points_rotated


if __name__ == '__main__':
    # Example usage:
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])
