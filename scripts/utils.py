import numpy as np

### COMPUTED TORQUE CONTROL 
def elementary_rotation_matrix_y(pitch):
    c = np.cos(pitch)
    s = np.sin(pitch)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ])

def rotation_matrix_to_axis(R):
    """
    Compute the rotation axis of the angle-axes rapresentation from a rotation matrix.

    Parameters
    ----------
    R : ndarray     3x3 rotation matrix

    Returns
    -------
    axis : ndarray  3D unit vector representing the rotation axis
    """

    axis_2 = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis_2 / 2
    return axis