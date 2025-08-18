#!/usr/bin/env python3

"""
Script to simulate the dynamics of the robot.

Steps:
1. Compute torques with Pinocchio's RNEA given q, qd, qdd.
2. Use custom forwardDynamics to recompute qdd from those torques.
3. Compare the reconstructed qdd with the original input.
"""

from __future__ import print_function
import pinocchio as pin
import numpy as np
from pathlib import Path

from dynamics import forwardDynamics


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def clean_matrix(vec: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Zero-out small numerical noise in a vector/matrix."""
    vec = vec.copy()
    vec[np.abs(vec) < tol] = 0.0
    return vec



# -----------------------------------------------------------------------------
# Dynamics Test
# -----------------------------------------------------------------------------
def dynamics_test():
    # Load the URDF model
    current_dir = Path(__file__).resolve().parent
def dynamics_test():
    # Load the URDF model
    current_dir = Path(__file__).resolve().parent
    urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")

    # Initialize Pinocchio model and data
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()


    # Define casual joint state
    q = np.array([1.36, -0.45, 0.40, 0.35, 1.01])  # Joint positions
    qd = np.array([0.1, 0.2, 0.0, 0.4, 0.3])       # Joint velocities
    qdd = np.array([1.0, 6.0, 3.0, 2.0, 1.0])      # Joint accelerations

    print(f"Joint positions (q): {q}")
    print(f"Joint velocities (qd): {qd}")
    print(f"Joint accelerations (qdd): {qdd}")

    # Compute the torques with RNEA
    tau = pin.rnea(model, data, q, qd, qdd)
    print(f"\nComputed joint torques (RNEA): {tau}")

    # Recompute accelerations with forward dynamics
    qdd_computed = forwardDynamics(q, qd, tau, model, data)
    print(f"Simulated joint accelerations (FD): {qdd_computed}")

    # Compare
    diff = clean_matrix(qdd_computed - qdd)
    error_norm = np.linalg.norm(diff)

    print(f"\nDifference (qdd_computed - qdd): {diff}")
    print(f"Error norm: {error_norm:.2e}")

    return diff, error_norm


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"\n{'='*40}")
    print("DYNAMIC SIMULATION")
    print(f"{'='*40}\n")
    diff, error_norm = dynamics_test()

    if np.allclose(diff, 0, atol=1e-6):
        print("✅ Dynamics test passed (differences within tolerance).")
    else:
        print("⚠️ Dynamics test shows non-negligible differences.")


if __name__ == "__main__":
    main()