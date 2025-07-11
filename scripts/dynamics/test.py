#!/usr/bin/env python3

from __future__ import print_function
import pinocchio as pin
import numpy as np
from pathlib import Path

from dynamics import forwardDynamics

# Load the URDF model
current_dir = Path(__file__).resolve().parent
urdf_path = current_dir.parent.parent / "urdf" / "giraffe_processed.urdf"
if not urdf_path.exists():
    raise FileNotFoundError(f"URDF file not found at {urdf_path}")

# Initialize Pinocchio model and data
model = pin.buildModelFromUrdf(str(urdf_path))
data = model.createData()

# Define the joint configuration for the RNEA computation
q = np.array([1.36, -0.45, 0.40, 0.35, 1.01])  # Joint positions
qd = np.array([0.1, 0.2, 0.0, 0.4, 0.3])       # Joint velocities
qdd = np.array([1.0, 6.0, 3.0, 2.0, 1.0])      # Joint accelerations

# Compute the torques using the RNEA method
taup = pin.rnea(model, data, q, qd, qdd)
print("Computed joint torques (RNEA):", taup)

# Compute bacward the joint accelaration 
qdd_computed = forwardDynamics(q, qd, taup, model, data)
print("Computed joint accelerations:", qdd_computed)

# Compute the difference between the computed and expected accelerations
diff = qdd_computed - qdd
print("Difference between computed and expected accelerations:", diff)

