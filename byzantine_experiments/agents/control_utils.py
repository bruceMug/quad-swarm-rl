"""
Direction -> action mapping for goal-directed Byzantine agents (Tier 2B / 2D).

The env's action space is four normalised motor thrusts, so an attacker that wants to *fly
somewhere* needs a position controller. Rather than inventing one, this replicates the
simulator's own Mellinger controller,
`gym_art/quadrotor_multi/quadrotor_control.py::NonlinearPositionController.step`
(lines 282-329), verbatim in its maths and gains.

Why replicate rather than import: that class cannot be instantiated in this environment.
Its `__init__` does `import tensorflow as tf` unconditionally (quadrotor_control.py:256)
even when `tf_control=False`, and tensorflow is not installed in `swarm_r311_recovered`.
The helper functions it relies on (`quadrotor_jacobian`, `normalize`, `cross`, `clamp_norm`)
*are* importable, so those are used directly and only the ~20 lines of controller body are
reproduced here.

Output convention. The Mellinger controller emits motor commands in [0, 1], but the env's
action space is `RawControl` with `zero_action_middle=True`, i.e. Box(-1, 1) mapped to
thrust by `0.5 * (a + 1)` (quadrotor_control.py:37-56). So the final step is the inverse of
that map, `a = 2 * thrust - 1`, which is what `thrust_to_action` does.
"""

import numpy as np

from gym_art.quadrotor_multi.quad_utils import clamp_norm, cross, normalize

GRAV = 9.81

# Gains copied from NonlinearPositionController.__init__ (quadrotor_control.py:266-267).
KP_P, KD_P = 4.5, 3.5
KP_A, KD_A = 200.0, 50.0

# `self.rot_des = np.eye(3)`, so xc_des is its first column.
XC_DES = np.array([1.0, 0.0, 0.0])

MAX_POS_ERROR = 4.0   # the `clamp_norm(to_goal, 4.0)` in the original


def _vee(R):
    """so(3) -> R^3, as defined inline in the original controller."""
    return np.array([R[2, 1], R[0, 2], R[1, 0]])


def thrust_to_action(thrusts):
    """[0, 1] motor commands -> [-1, 1] env actions (inverse of RawControl's affine map)."""
    return 2.0 * np.asarray(thrusts, dtype=np.float64) - 1.0


def action_to_thrust(action):
    return 0.5 * (np.asarray(action, dtype=np.float64) + 1.0)


def position_control_action(pos, vel, rot, omega, target, jacobian_inv):
    """
    Motor action that flies a drone from `pos` toward `target`.

    Args:
        pos, vel, omega: (3,) current state, world frame.
        rot: (3, 3) body-to-world rotation matrix.
        target: (3,) desired position.
        jacobian_inv: inverse of `quadrotor_jacobian(dynamics)` for this drone.

    Returns an action in [-1, 1]^4, already clipped.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    to_goal = np.asarray(target, dtype=np.float64) - pos
    e_p = -clamp_norm(to_goal, MAX_POS_ERROR)
    e_v = vel
    acc_des = -KP_P * e_p - KD_P * e_v + np.array([0.0, 0.0, GRAV])

    zb_des, _ = normalize(acc_des)
    yb_des, _ = normalize(cross(zb_des, XC_DES))
    xb_des = cross(yb_des, zb_des)
    R_des = np.column_stack((xb_des, yb_des, zb_des))

    e_R = 0.5 * _vee(R_des.T @ rot - rot.T @ R_des)
    e_R[2] *= 0.2                      # "slow down yaw dynamics", as in the original
    e_w = omega

    dw_des = -KP_A * e_R - KD_A * e_w
    thrust_mag = float(np.dot(acc_des, rot[:, 2]))

    des = np.append(thrust_mag, dw_des)
    thrusts = np.clip(jacobian_inv @ des, 0.0, 1.0)
    return np.clip(thrust_to_action(thrusts), -1.0, 1.0)


def jacobians_for(dynamics):
    """
    (J, J^-1) for one drone, from its own dynamics parameters.

    `quadrotor_jacobian` (quadrotor_control.py:158) builds
        J = [[thrust_max/mass ...],      -> row 0: collective vertical acceleration
             [torque rows / inertia]]    -> rows 1..3: angular acceleration about x, y, z
    so `J @ thrusts` decomposes four motor commands into
    `[collective_accel, roll_accel, pitch_accel, yaw_accel]` -- the body-axis basis Tier 2A
    needs in order to flip roll/pitch while leaving collective thrust alone.
    """
    from gym_art.quadrotor_multi.quadrotor_control import quadrotor_jacobian
    J = quadrotor_jacobian(dynamics)
    return J, np.linalg.inv(J)


def jacobian_inv_for(dynamics):
    """Inverse control Jacobian only (kept for callers that do not need J)."""
    return jacobians_for(dynamics)[1]


def action_to_body(action, jacobian):
    """[-1,1]^4 motor action -> [collective, roll, pitch, yaw] accelerations."""
    return jacobian @ action_to_thrust(action)


def body_to_action(body, jacobian_inv):
    """[collective, roll, pitch, yaw] -> [-1,1]^4 motor action, clipped legally."""
    thrusts = np.clip(jacobian_inv @ np.asarray(body, dtype=np.float64), 0.0, 1.0)
    return np.clip(thrust_to_action(thrusts), -1.0, 1.0)
