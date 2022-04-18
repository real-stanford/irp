import numpy as np
import mujoco_py as mj

class MujocoCompensatedPDController:
    """
    Compensated joint-space PD controller
    """
    def __init__(self, sim, joint_names, kp=1, kv=1, use_sim_state=True):
        self.sim = sim
        self.model = sim.model
        self.use_sim_state = use_sim_state
        self.kp = kp
        self.kv = kv

        sim.forward()
        model = sim.model
        # assums 1DoF for all joints
        joint_ids = [
            model.joint_name2id(x) 
            for x in joint_names]

        # compute joint data addrs
        self.joint_pos_addrs = [model.get_joint_qpos_addr(name) for name in joint_names]
        self.joint_vel_addrs = [model.get_joint_qvel_addr(name) for name in joint_names]

        joint_pos_addrs = []
        for elem in self.joint_pos_addrs:
            if isinstance(elem, tuple):
                joint_pos_addrs += list(range(elem[0], elem[1]))
            else:
                joint_pos_addrs.append(elem)
        self.joint_pos_addrs = joint_pos_addrs

        joint_vel_addrs = []
        for elem in self.joint_vel_addrs:
            if isinstance(elem, tuple):
                joint_vel_addrs += list(range(elem[0], elem[1]))
            else:
                joint_vel_addrs.append(elem)
        self.joint_vel_addrs = joint_vel_addrs

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = 0
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(model.jnt_type):
            if ii in joint_ids:
                self.joint_dyn_addrs.append(index)
                if joint_type == 0:  # free joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                    index += 6  # derivative has 6 dimensions
                elif joint_type == 1:  # ball joint
                    self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                    index += 3  # derivative has 3 dimension
                else:  # slide or hinge joint
                    index += 1  # derivative has 1 dimensions

        self.joint_pos_addrs = np.copy(self.joint_pos_addrs)
        self.joint_vel_addrs = np.copy(self.joint_vel_addrs)
        self.joint_dyn_addrs = np.copy(self.joint_dyn_addrs)

        # number of controllable joints in the robot arm
        self.N_JOINTS = len(self.joint_dyn_addrs)
        # number of joints in the Mujoco simulation
        N_ALL_JOINTS = self.sim.model.nv

        # need to calculate the joint_dyn_addrs indices in flat vectors returned
        # for the Jacobian
        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_JOINTS
            [self.joint_dyn_addrs + (ii * N_ALL_JOINTS) for ii in range(3)]
        )

        # for the inertia matrix
        self.M_indices = [
            ii * N_ALL_JOINTS + jj
            for jj in self.joint_dyn_addrs
            for ii in self.joint_dyn_addrs
        ]

        # a place to store data returned from Mujoco
        self._g = np.zeros(self.N_JOINTS)
        self._J3NP = np.zeros(3 * N_ALL_JOINTS)
        self._J3NR = np.zeros(3 * N_ALL_JOINTS)
        self._J6N = np.zeros((6, self.N_JOINTS))
        self._MNN_vector = np.zeros(N_ALL_JOINTS ** 2)
        self._MNN = np.zeros(self.N_JOINTS ** 2)
        self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        self._x = np.ones(4)
        self.N_ALL_JOINTS = N_ALL_JOINTS

    @property
    def q(self):
        return np.copy(self.sim.data.qpos[self.joint_pos_addrs])
    
    @property
    def dq(self):
        return np.copy(self.sim.data.qvel[self.joint_vel_addrs])

    def generate(self, target, target_velocity=None):
        if target_velocity is None:
            target_velocity = np.zeros(self.N_JOINTS)

        q = self.q
        dq = self.dq

        q_tilde = target - q
        if False:
            # calculate the direction for each joint to move, wrapping
            # around the -pi to pi limits to find the shortest distance
            q_tilde = ((target - q + np.pi) % (np.pi * 2)) - np.pi

        # get the joint space inertia matrix
        M = self.M(q)
        u = np.dot(M, (self.kp * q_tilde + self.kv * (target_velocity - dq)))
        # account for gravity
        u -= self.g(q)
        return u
    
    def send_forces(self, u, step=False):
        # NOTE: the qpos_addr's are unrelated to the order of the motors
        # NOTE: assuming that the robot arm motors are the first len(u) values
        self.sim.data.ctrl[:] = u[:]

        # move simulation ahead one time step
        if step:
            self.sim.step()

    def g(self, q=None):
        """Returns qfrc_bias variable, which stores the effects of Coriolis,
        centrifugal, and gravitational forces

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        # TODO: For the Coriolis and centrifugal functions, setting the
        # velocity before calculation is important, how best to do this?
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        g = -1 * self.sim.data.qfrc_bias[self.joint_dyn_addrs]

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)
        return g
    
    def M(self, q=None):
        """Returns the inertia matrix in task space

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        if not self.use_sim_state and q is not None:
            old_q, old_dq, old_u = self._load_state(q)

        # stored in mjData.qM, stored in custom sparse format,
        # convert qM to a dense matrix with mj_fullM
        mj.cymj._mj_fullM(self.model, self._MNN_vector, self.sim.data.qM)
        M = self._MNN_vector[self.M_indices]
        M = M.reshape((self.N_JOINTS, self.N_JOINTS))

        if not self.use_sim_state and q is not None:
            self._load_state(old_q, old_dq, old_u)

        return np.copy(M)

    def _load_state(self, q=None, dq=None, u=None):
        """Change the current joint angles

        Parameters
        ----------
        q: np.array
            The set of joint angles to move the arm to [rad]
        dq: np.array
            The set of joint velocities to move the arm to [rad/sec]
        u: np.array
            The set of joint forces to apply to the arm joints [Nm]
        """
        # save current state
        old_q = np.copy(self.sim.data.qpos[self.joint_pos_addrs])
        old_dq = np.copy(self.sim.data.qvel[self.joint_vel_addrs])
        old_u = np.copy(self.sim.data.ctrl)

        # update positions to specified state
        if q is not None:
            self.sim.data.qpos[self.joint_pos_addrs] = np.copy(q)
        if dq is not None:
            self.sim.data.qvel[self.joint_vel_addrs] = np.copy(dq)
        if u is not None:
            self.sim.data.ctrl[:] = np.copy(u)

        # move simulation forward to calculate new kinamtic information
        self.sim.forward()

        return old_q, old_dq, old_u
