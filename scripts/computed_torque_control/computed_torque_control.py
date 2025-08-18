import numpy as np
import pinocchio as pin

from ..kinematics.kinematics import geometric2analyticJacobian


def task_space_torque_control(
    model, data, q, dq,
    x, dx, 
    x_des, dx_des, ddx_des,
    Kp=None,
    Kd=None,
    torque_limit=None,
    eps_reg=1e-6,
    fd_eps=1e-6
):
    """
    Task-space inverse dynamics controller (computed torque / operational space control)
    for a 4D task space: [x, y, z, pitch].

    Parameters
    ----------
    model : pin.Model   Pinocchio robot model
    data : pin.Data     Pinocchio data container
    q : ndarray         Joint positions (n,)
    dq : ndarray        Joint velocities (n,)

    x: ndarray         Current task position from feedback loop (3D position + pitch) (4,) 
    dx: ndarray        Current task velocity from feedback loop (4,)

    x_des : ndarray     Desired task position (3D position + pitch) (4,)
    dx_des : ndarray    Desired task velocity (4,)
    ddx_des : ndarray   Desired task acceleration (4,)

    Kp : ndarray        Task-space proportional gain (4x4)
    Kd : ndarray        Task-space derivative gain (4x4)

    torque_limit: if not None, clip torques elementwise to ±torque_limit
    eps_reg: small regularization added to A if it is singular/ill-conditioned
    fd_eps: finite diff epsilon for Jdot estimation

    Returns
    -------
    tau : ndarray       Joint torques (n,)

    Notes / assumptions:
      - "pitch" is defined as the second Euler angle returned by pin.rpy.matrixToRpy(R),
        i.e. roll, pitch, yaw (rotation around X, Y, Z respectively). We consistently use
        the analytic Jacobian (geometric->analytic) so the angular rows correspond to
        Euler angle rates [r_dot, p_dot, y_dot] and the pitch row is index 1 (0-based).
      - If the analytic Jacobian time-variation (Jdot) is not available from Pinocchio,
        a small finite-difference approximation can be used (controlled by
        use_finite_diff_jdot flag). Finite-diff is slower and less accurate but safe.
    """

    # set default gains if not provided
    if Kp is None:
        Kp = np.diag([100., 100., 100., 50.])
    if Kd is None:
        Kd = np.diag([20., 20., 20., 10.])

    # End Effector Current State
    frame_id = model.getFrameId("mic")
    oMf = data.oMf[frame_id]
    T_0e = oMf.homogeneous

    # analytic Jacobian (6 x nv) maps qdot -> [x_dot; euler_rates]
    J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ja = geometric2analyticJacobian(J6, T_0e)  # 6 x nv (first 3 linear, next 3 Euler rates in same rpy order)

    # Build reduced task Jacobian Jx (4 x nv): linear(3) + pitch row (the second Euler rate row)
    nv = Ja.shape[1]
    Jx = np.zeros((4, nv))
    Jx[0:3, :] = Ja[0:3, :]              # x,y,z rows   linear part
    Jx[3, :] = Ja[4, :]                  # pitch row    euler part

    # errors and commanded acceleration (computed-torque style)
    ex = x_des - x
    edx = dx_des - dx
    ddx_cmd = ddx_des + Kd @ edx + Kp @ ex

    # --- 2. Dynamics: mass matrix and bias term ---
    M = pin.crba(model, data, q)                            # inertia matrix 
    h = pin.rnea(model, data, q, dq, np.zeros_like(dq))     # bias (C*qdot + g)

    # invert M, if M is not invertible, use pseudo-inverse (NB: M usualy is a square matrix nxn )
    try:
        M_inv = np.linalg.inv(M)        
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    # Task Space inertia (A = J M^{-1} J^T)
    A = Jx @ M_inv @ Jx.T

    # Regularize A if near singular or badly conditioned
    cond_A = np.linalg.cond(A)
    if np.isnan(cond_A) or cond_A > 1e12:
        # add a bit larger regularization in extreme cases
        A += max(eps_reg, 1e-3) * np.eye(4)
    elif cond_A > 1e6:
        A += eps_reg * np.eye(4)

    # Lambda (task-space inertia)
    try:
        Lambda = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Lambda = np.linalg.pinv(A)

    # --- Compute Jdot * dq ---
    # Preferred: if pin provides a function to get frame jacobian time variation, use it.
    # Many pinocchio python builds include: pin.getFrameJacobianTimeVariation(model, data, frame_id, ref)
    Jdot_dq_task = None
    
    J6dot = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # transform J6dot (6xnv) to analytic version consistent with Ja
    # compute analytic time-variation approximately by mapping the angular rows similarly to geometric2analyticJacobian
    # (here we reuse the same geometric2analyticJacobian mapping for J6dot using the current T_0e)
    Ja_dot = geometric2analyticJacobian(J6dot, T_0e)
    jdot_full = Ja_dot @ dq   # 6-vector of [linear_ddot_contrib; euler_rates_dot_contrib] where contrib=Jdot*dq
    Jdot_dq_task = np.array([jdot_full[0], jdot_full[1], jdot_full[2], jdot_full[4]])

    # Task-space bias (mu) and commanded task-space force Fx 
    # mu = Lambda (J * M^{-1} * h - Jdot * qdot)
    mu = Lambda @ (Jx @ (M_inv @ h) - Jdot_dq_task)

    # Desired task-space force
    Fx = Lambda @ ddx_cmd + mu

    # --- 3. Map to joint torques ---
    tau = Jx.T @ Fx

    # Optional: torque limits
    if torque_limit is not None:
        tau = np.clip(tau, -abs(torque_limit), abs(torque_limit))

    return tau



def pd_gains_for_settling_time(Ts=7.0, zeta=1.2):

    # Guadagni più piccoli per una risposta più lenta e morbida
    Kp = np.diag([3.5, 3.5, 3.5, 8.0])
    Kd_vals = 2*np.array([1.2, 1.2, 1.2, 0.25])*np.sqrt(Kp.diagonal())
    Kd = np.diag(Kd_vals)

    return Kp, Kd




def task_space_torque_control_with_secondary_task(
    model, data, q, dq,
    x, dx,
    x_des, dx_des, ddx_des,
    Kp=None,
    Kd=None,
    torque_limit=None,
    eps_reg=1e-6,
    fd_eps=1e-6,

    q0=None,                # desired joint configuration in null space
    Kq=None,                # joint-space proportional gain
    Dq=None                 # joint-space derivative gain
):
    """
    Task-space inverse dynamics controller (computed torque / operational space control)
    for a 4D task space: [x, y, z, pitch].

    Parameters
    ----------
    model : pin.Model   Pinocchio robot model
    data : pin.Data     Pinocchio data container
    q : ndarray         Joint positions (n,)
    dq : ndarray        Joint velocities (n,)

    x: ndarray         Current task position from feedback loop (3D position + pitch) (4,) 
    dx: ndarray        Current task velocity from feedback loop (4,)

    x_des : ndarray     Desired task position (3D position + pitch) (4,)
    dx_des : ndarray    Desired task velocity (4,)
    ddx_des : ndarray   Desired task acceleration (4,)

    Kp : ndarray        Task-space proportional gain (4x4)
    Kd : ndarray        Task-space derivative gain (4x4)

    torque_limit: if not None, clip torques elementwise to ±torque_limit
    eps_reg: small regularization added to A if it is singular/ill-conditioned
    fd_eps: finite diff epsilon for Jdot estimation

    q0=None,            desired joint configuration in null space
    Kq=None,            joint-space proportional gain
    Dq=None             joint-space derivative gain


    Returns
    -------
    tau : ndarray       Joint torques (n,)

    Notes / assumptions:
      - "pitch" is defined as the second Euler angle returned by pin.rpy.matrixToRpy(R),
        i.e. roll, pitch, yaw (rotation around X, Y, Z respectively). We consistently use
        the analytic Jacobian (geometric->analytic) so the angular rows correspond to
        Euler angle rates [r_dot, p_dot, y_dot] and the pitch row is index 1 (0-based).
      - If the analytic Jacobian time-variation (Jdot) is not available from Pinocchio,
        a small finite-difference approximation can be used (controlled by
        use_finite_diff_jdot flag). Finite-diff is slower and less accurate but safe.
    """

    # set default gains if not provided
    if Kp is None:
        Kp = np.diag([100., 100., 100., 50.])
    if Kd is None:
        Kd = np.diag([20., 20., 20., 10.])

    # End Effector Current State
    frame_id = model.getFrameId("mic")
    oMf = data.oMf[frame_id]
    T_0e = oMf.homogeneous

    # analytic Jacobian (6 x nv) maps qdot -> [x_dot; euler_rates]
    J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ja = geometric2analyticJacobian(J6, T_0e)  # 6 x nv (first 3 linear, next 3 Euler rates in same rpy order)

    # Build reduced task Jacobian Jx (4 x nv): linear(3) + pitch row (the second Euler rate row)
    nv = Ja.shape[1]
    Jx = np.zeros((4, nv))
    Jx[0:3, :] = Ja[0:3, :]              # x,y,z rows   linear part
    Jx[3, :] = Ja[4, :]                  # pitch row    euler part

    # errors and commanded acceleration (computed-torque style)
    ex = x_des - x
    edx = dx_des - dx
    ddx_cmd = ddx_des + Kd @ edx + Kp @ ex


    # --- 2. Dynamics: mass matrix and bias term ---
    M = pin.crba(model, data, q)                            # inertia matrix 
    h = pin.rnea(model, data, q, dq, np.zeros_like(dq))     # bias (C*qdot + g)

    # invert M, if M is not invertible, use pseudo-inverse (NB: M usualy is a square matrix nxn )
    try:
        M_inv = np.linalg.inv(M)        
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    # Task Space inertia (A = J M^{-1} J^T)
    A = Jx @ M_inv @ Jx.T

    # Regularize A if near singular or badly conditioned
    cond_A = np.linalg.cond(A)
    if np.isnan(cond_A) or cond_A > 1e12:
        # add a bit larger regularization in extreme cases
        A += max(eps_reg, 1e-3) * np.eye(4)
    elif cond_A > 1e6:
        A += eps_reg * np.eye(4)

    # Lambda (task-space inertia)
    try:
        Lambda = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Lambda = np.linalg.pinv(A)

    # --- Compute Jdot * dq ---
    # Preferred: if pin provides a function to get frame jacobian time variation, use it.
    # Many pinocchio python builds include: pin.getFrameJacobianTimeVariation(model, data, frame_id, ref)
    Jdot_dq_task = None
    
    J6dot = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # transform J6dot (6xnv) to analytic version consistent with Ja
    # compute analytic time-variation approximately by mapping the angular rows similarly to geometric2analyticJacobian
    # (here we reuse the same geometric2analyticJacobian mapping for J6dot using the current T_0e)
    Ja_dot = geometric2analyticJacobian(J6dot, T_0e)
    jdot_full = Ja_dot @ dq   # 6-vector of [linear_ddot_contrib; euler_rates_dot_contrib] where contrib=Jdot*dq
    Jdot_dq_task = np.array([jdot_full[0], jdot_full[1], jdot_full[2], jdot_full[4]])

    # Task-space bias (mu) and commanded task-space force Fx ---
    # mu = Lambda (J * M^{-1} * h - Jdot * qdot)
    mu = Lambda @ (Jx @ (M_inv @ h) - Jdot_dq_task)

    # Desired task-space force
    Fx = Lambda @ ddx_cmd + mu

    # --- 3. Main task-space torque ---
    tau_task = Jx.T @ Fx


    # --- 4. Null-space control ---
    if q0 is not None:
        if Kq is None:
            Kq = np.eye(len(q)) * 5.0
        if Dq is None:
            Dq = np.eye(len(q)) * 1.0

        # Joint-space PD toward q0
        tau_q = Kq @ (q0 - q) - Dq @ dq

        # Null-space projector: N = [ I - Jx^T * Jx^T#] 
        N = np.eye(len(q)) - Jx.T @ (Lambda @ (Jx @ M_inv))

        # Add null-space torque
        tau = tau_task + N @ tau_q
    else:
        tau = tau_task


    # Optional: torque limits
    if torque_limit is not None:
        tau = np.clip(tau, -abs(torque_limit), abs(torque_limit))

    return tau
