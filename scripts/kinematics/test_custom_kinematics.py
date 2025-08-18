#!/usr/bin/env python3

"""
Script to compare a custom kinematics implementation with Pinocchio results.

Features:
- Direct kinematics comparison
- Differential kinematics comparison (Jacobian)
- Inverse kinematics comparison

"""

from pathlib import Path

import numpy as np
import pinocchio as pin

from kinematics import directKinematics, differentKinematics, inverseKinematics, rot2rpy




# -----------------------------------------------------------------------------
# Load URDF and initialize Pinocchio model
# -----------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
if not urdf_path.exists():
    raise FileNotFoundError(f"URDF file not found at {urdf_path}")

model = pin.buildModelFromUrdf(str(urdf_path))
data = model.createData()

# Frames of interest
FRAMES = ["base_link", "shoulder_link", "arm_link",
          "extend_link", "wrist_link", "mic_link", "mic"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_frame_transform(frame: str, q: np.ndarray) -> np.ndarray:
    """Return the homogeneous transform of a frame using Pinocchio."""
    try:
        frame_id = model.getFrameId(frame)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[frame_id].homogeneous
    except Exception as e:
        print(f"Frame '{frame}' not found: {e}")
        return None


def clean_matrix(mat: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Zero-out small numerical noise in a matrix."""
    mat = mat.copy()
    mat[np.abs(mat) < tol] = 0.0
    return mat


# -----------------------------------------------------------------------------
# Direct Kinematics Tests
# -----------------------------------------------------------------------------
def direct_kinematics_single():
    """
    Run a single DK comparison with random configuration.
    """
    q = pin.randomConfiguration(model)
    print(f"Random joint configuration (q): {q}")

    T_custom_list = directKinematics(q)

    for i, frame in enumerate(FRAMES):
        T_pin = get_frame_transform(frame, q)
        if T_pin is None:
            continue
        print(f"Frame: {frame}")
        print(f"- Custom T:\n{T_custom_list[i]}")
        print(f"- Pinocchio T:\n{T_pin}")


def direct_kinematics_avg_error(n_iter: int = 10):
    """
    Run multiple DK comparisons and return average element-wise error per frame.
    """
    accumulated = {f: np.zeros((4, 4)) for f in FRAMES}
    counts = {f: 0 for f in FRAMES}

    for _ in range(n_iter):
        q = pin.randomConfiguration(model)
        T_custom_list = directKinematics(q)
        for i, frame in enumerate(FRAMES):
            T_pin = get_frame_transform(frame, q)
            if T_pin is None:
                continue
            delta = clean_matrix(T_custom_list[i] - T_pin)
            accumulated[frame] += delta
            counts[frame] += 1

    print(f"Average DK Errors over {n_iter} iterations")
    avg_errors = {}
    for frame in FRAMES:
        if counts[frame] > 0:
            avg = clean_matrix(accumulated[frame] / counts[frame])
            avg_errors[frame] = avg
            print(f"\nFrame: {frame}\n{avg}")
        else:
            avg_errors[frame] = None
            print(f"Frame '{frame}' not found in any iteration")

    return avg_errors




# -----------------------------------------------------------------------------
# Differential Kinematics Tests
# -----------------------------------------------------------------------------
def differential_kinematics_single(frame: str = "mic"):
    """
    Compare custom and Pinocchio Jacobians for one frame.
    """
    q = pin.randomConfiguration(model)
    J_custom = differentKinematics(q)

    try:
        frame_id = model.getFrameId(frame)
        J_pin = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    except Exception as e:
        print(f"Could not compute Jacobian for '{frame}': {e}")
        return

    diff = J_custom - J_pin
    diff = clean_matrix(diff)

    print(f"Frame: {frame}")
    print(f"- Custom J:\n{J_custom}\n")
    print(f"- Pinocchio J:\n{J_pin}\n")
    print(f"- Difference:\n{diff}")




# -----------------------------------------------------------------------------
# Inverse Kinematics Tests
# -----------------------------------------------------------------------------
def inverse_kinematics_single(frame: str = "mic"):
    """
    Validate custom IK by generating a target from random q and solving for it.
    """
    q_true = np.random.uniform(-np.pi, np.pi, model.nq)
    print(f"Ground truth joint configuration: {q_true}")

    target_pose = get_frame_transform(frame, q_true)
    if target_pose is None:
        return

    desired_pos = target_pose[:3, 3]
    desired_rot = target_pose[:3, :3]
    desired_pitch = rot2rpy(desired_rot)[1]

    print(f"Target position from FK:\n{desired_pos}")
    print(f"Target pitch from FK: {desired_pitch}")

    q_ik = inverseKinematics(desired_pos, desired_pitch, model, data)
    print(f"Inverse kinematics solution q:\n{q_ik}")

    # Compare poses
    result_pose = get_frame_transform(frame, q_ik)
    if result_pose is None:
        return
    
    result_pos = result_pose[:3, 3]
    result_rot = result_pose[:3, :3]
    result_pitch = rot2rpy(result_rot)[1]

    print(f"Result position:\n{result_pos}")
    print(f"Result pitch: {result_pitch}")

    print(f"Position error: {np.linalg.norm(result_pos - desired_pos)}")
    print(f"Pitch error: {result_pitch - desired_pitch}")




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"\n{'='*40}")
    print("DIRECT KINEMATICS TEST")
    print(f"{'='*40}\n")
    direct_kinematics_avg_error(n_iter=10)

    print(f"\n{'='*40}")
    print("DIFFERENTIAL KINEMATICS TEST")
    print(f"{'='*40}\n")
    differential_kinematics_single(frame="mic")

    print(f"\n{'='*40}")
    print("INVERSE KINEMATICS TEST")
    print(f"{'='*40}\n")
    inverse_kinematics_single(frame="mic")


if __name__ == "__main__":
    main()


