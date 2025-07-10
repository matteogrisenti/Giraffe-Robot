#!/usr/bin/env python3

# This script compares a custom kinematics implementation with the Pinocchio library's result

from __future__ import print_function
import pinocchio as pin
import numpy as np
from pathlib import Path

from direct_kinematics import directKinematics  

# Load the URDF model
current_dir = Path(__file__).resolve().parent
urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
if not urdf_path.exists():
    raise FileNotFoundError(f"URDF file not found at {urdf_path}")

# Initialize Pinocchio model and data
model = pin.buildModelFromUrdf(str(urdf_path))
data = model.createData()

# Define the frames of interest in the kinematic chain
frames = ['base_link', 'shoulder_link', 'arm_link', 'extend_link', 'wrist_link', 'mic_link', 'mic']


def single_comparison():
    """    
    Runs a single comparison between custom and Pinocchio forward kinematics.
    This function generates a random joint configuration, computes the forward kinematics using both methods,
    and prints the resulting transformation matrices for each frame.
    """
    # Define a test joint configuration of appropriate size
    # test_q = np.zeros(model.nq)   ->  Used to first testing

    # Generate a random valid joint configuration
    test_q = pin.randomConfiguration(model)
    print("Test joint configuration (q):", test_q, "\n")


    # Run custom forward kinematics
    T_w0, T_w1, T_w2, T_w3, T_w4, T_w5, T_we = directKinematics(test_q)
    custom_T = [T_w0, T_w1, T_w2, T_w3, T_w4, T_w5, T_we]

    # Run Pinocchio forward kinematics
    pin.forwardKinematics(model, data, test_q)
    pin.updateFramePlacements(model, data)

    for frame in frames:
        try:
            ee_id = model.getFrameId(frame)
            pin_T_0x = data.oMf[ee_id].homogeneous
        except:
            print(f"Frame '{frame}' not found in model. Available frames: {[f.name for f in model.frames]}")

        # Compare results
        print(f"\nFrame: {frame}")
        print("- Custom T:\n", custom_T[frames.index(frame)])
        print("-Pinocchio T:\n", pin_T_0x)


def test_DK_iteration():
    """
    Runs a single DK test and returns per-frame matrix errors (custom - pinocchio) as 4x4 numpy arrays.
    """
    q = pin.randomConfiguration(model)
    T_custom_list = directKinematics(q)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    errors = {}
    for i, frame in enumerate(frames):
        try:
            frame_id = model.getFrameId(frame)
            T_pin = data.oMf[frame_id].homogeneous
            T_custom = T_custom_list[i]
            delta = T_custom - T_pin
            delta[np.abs(delta) < 1e-4] = 0.0  # Clean small numerical noise
            errors[frame] = delta
        except:
            print(f"[Warning] Frame '{frame}' not found in model.")
            continue

    return errors


def test_DK(n_iterations=10):
    """
    Runs DK tests for multiple iterations and returns the average element-wise 4x4 error matrix per frame.
    """
    accumulated_errors = {frame: np.zeros((4, 4)) for frame in frames}
    valid_counts = {frame: 0 for frame in frames}

    for _ in range(n_iterations):
        iteration_errors = test_DK_iteration()
        for frame, err_matrix in iteration_errors.items():
            accumulated_errors[frame] += err_matrix
            valid_counts[frame] += 1

    avg_errors = {}
    print(f"\n=== Average Element-wise Error Matrices over {n_iterations} iterations ===")
    for frame in frames:
        if valid_counts[frame] > 0:
            avg = accumulated_errors[frame] / valid_counts[frame]
            avg[np.abs(avg) < 1e-4] = 0.0  # Clean again final results
            avg_errors[frame] = avg
            print(f"\nFrame: {frame}\n{avg}")
        else:
            avg_errors[frame] = None
            print(f"\nFrame: {frame} -> [Frame not found in model]")

    return avg_errors


if __name__ == "__main__":
    print("Running single comparison...")
    single_comparison()

    print("\nRunning DK test with 10 iterations...")
    test_DK(n_iterations=10)

