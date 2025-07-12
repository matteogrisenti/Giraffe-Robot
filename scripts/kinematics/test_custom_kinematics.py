#!/usr/bin/env python3

# This script compares a custom kinematics implementation with the Pinocchio library's result

from __future__ import print_function
import pinocchio as pin
import numpy as np
from pathlib import Path

from kinematics import directKinematics, differentKinematics, inverseKinematics, rot2rpy

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



######################################### DIRECT KINEMATICS TEST #########################################
def direct_kinematic_single_comparison():
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
        print("- Pinocchio T:\n", pin_T_0x)


def test_direct_kinematic_iteration():
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


def test_direct_kinematic(n_iterations=10):
    """
    Runs DK tests for multiple iterations and returns the average element-wise 4x4 error matrix per frame.
    """
    accumulated_errors = {frame: np.zeros((4, 4)) for frame in frames}
    valid_counts = {frame: 0 for frame in frames}

    for _ in range(n_iterations):
        iteration_errors = test_direct_kinematic_iteration()
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



######################################### DIFFERNTIAL KINEMATICS TEST #########################################
def differential_kinematic_single_comparison():
    # Generate a random valid joint configuration
    test_q = pin.randomConfiguration(model)
    print("Test joint configuration (q):", test_q, "\n")

    J_custom = differentKinematics(test_q)  

    # Get the frame index (use frame name or index directly)
    frame_name = "mic"
    frame_id = model.getFrameId(frame_name)

    # Compute Pinocchio's Jacobian at that frame
    J_pinocchio = pin.computeFrameJacobian(model, data, test_q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # Compare results
    print("- Custom J:\n", J_custom)
    print("- Pinocchio J:\n", J_pinocchio)

    # Compute the difference
    jacobian_diff = J_custom - J_pinocchio
    print(f"Custom vs Pinocchio Jacobian difference at frame '{frame_name}':\n", jacobian_diff)




############################################# INVERSE KINEMATICS TEST #########################################
def inverse_kinematic_single_comparison():
    # Generate a random valid joint configuration
    test_q = pin.randomConfiguration(model)
    print("Test joint configuration (q):", test_q, "\n")

    # Derive the end-effector pose from the forward kinematics
    pin.forwardKinematics(model, data, test_q)
    pin.updateFramePlacements(model, data)

    ee_id = model.getFrameId("mic")
    target_pose = data.oMf[ee_id].homogeneous
    print("Target pose (end-effector):\n", target_pose)

    desired_position = target_pose[:3, 3]
    desired_rotation = target_pose[:3, :3]
    desired_pitch = rot2rpy(desired_rotation)[1]  # Convert rotation matrix to roll-pitch-yaw angles

    # Run custom inverse kinematics
    inverse_q = inverseKinematics(desired_position, desired_pitch, model, data)  # Assuming this function returns a joint configuration
    print("Inverse kinematics result (q):", inverse_q, "\n")

    # Compare results
    error = test_q - inverse_q
    print("Error between original and IK result:", error)



if __name__ == "__main__":
    # print("Running single comparison...")
    # direct_kinematic_single_comparison()

    # print("\nRunning DK test with 10 iterations...")
    # test_direct_kinematic(n_iterations=10)

    #print("\nRunning differential kinematic single comparison...")
    #differential_kinematic_single_comparison()

    print("\nRunning inverse kinematic single comparison...")
    inverse_kinematic_single_comparison()

