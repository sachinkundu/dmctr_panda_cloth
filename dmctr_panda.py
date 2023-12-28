"""Minimal working example of the dm_robotics Panda model."""
import time

import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import environment
from dm_robotics.panda import arm_constants
from dm_robotics.panda import run_loop, utils
from dm_robotics.panda import parameters as params
from dm_robotics.moma import entity_initializer, prop
from dm_robotics.moma.sensors import prop_pose_sensor
from dm_robotics.transformations import transformations as tr
from dm_control.composer.variation import distributions, rotations

import modern_robotics as mr

N = 50
Tf = 0.1 * (N - 1)

# class Cloth(prop.Prop):
#     """Simple cloth prop that consists of a MuJoco flexcomp."""
#
#     def _build(self, *args, **kwargs):
#         del args, kwargs
#         mjcf_root = mjcf.RootElement()
#
#         mjcf_root.extension.add('plugin', plugin="mujoco.elasticity.shell")
#
#         # Props need to contain a body called prop_root
#         mjcf_root.worldbody.add('body', name='prop_root')
#
#         cloth_object = mjcf.from_file('cloth.xml')
#         mjcf_root.attach(cloth_object)
#
#         super()._build('cloth', mjcf_root)


def rmat_to_omega_vector(rmat1: np.ndarray, rmat2: np.ndarray):
    A = rmat2 * rmat1.T
    theta = np.arccos((np.trace(A) - 1) * 0.5)
    W = (1/(2*0.1)) * (theta / np.sin(theta)) * (A - A.T)
    omega = mr.so3ToVec(W)
    return omega


class Agent:
    """The agent produces a trajectory tracing the path of an eight
    in the x/y control frame of the robot using end-effector velocities.
    """

    def __init__(self, spec: specs.BoundedArray) -> None:
        self._spec = spec
        self.xtrajectory = None
        self.xvtrajectory = []
        self.step_no = 0
        self.current_time = 0

    def calculate_trajectory(self, timestep: dm_env.TimeStep):
        t = 0.1
        block_pose = timestep.observation['block_pose']
        tcp_pose = timestep.observation['panda_tcp_pose']
        block_hmat = tr.pos_quat_to_hmat(block_pose[:3], block_pose[3:])
        tcp_hmat = tr.pos_quat_to_hmat(tcp_pose[:3], tcp_pose[3:])
        print(f"Initial block pos = {block_pose[:3]} -- Initial tcp_pose = {tcp_pose[:3]}")
        self.xtrajectory = mr.CartesianTrajectory(tcp_hmat, block_hmat, Tf, N+1, 5)

        for frame1, frame2 in zip(self.xtrajectory, self.xtrajectory[1:]):
            velocities = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
            omega = rmat_to_omega_vector(frame2[:-1, :3], frame1[:-1, :3])
            velocities[:3] = (frame2[:-1, -1] - frame1[:-1, -1]) / t
            velocities[3:-1] = omega
            self.xvtrajectory.append(velocities)

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:

        if not self.xtrajectory:
            self.current_time = time.time()
            self.calculate_trajectory(timestep)

        action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        action[-1] = 1

        if self.step_no < N:
            if self.step_no > 0:
                expected_tcp_pos = self.xtrajectory[self.step_no - 1][:-1, -1]
                actual_tcp_pos = timestep.observation['panda_tcp_pos']
                print(f"expected: {expected_tcp_pos} -- actual: {actual_tcp_pos}")
                # print(f"Action executed after {time.time() - self.current_time} s")
            else:
                print(f"Starting time: {self.current_time}")

            action = self.xvtrajectory[self.step_no]
            # print(f"action = {action}")
            self.step_no += 1
            self.current_time = time.time()

        return action


if __name__ == '__main__':
    # We initialize the default configuration for logging
    # and argument parsing. These steps are optional.
    utils.init_logging()
    parser = utils.default_arg_parser()
    args = parser.parse_args()

    # Use RobotParams to customize Panda robots added to the environment.
    robot_params = params.RobotParams(robot_ip=args.robot_ip, actuation=arm_constants.Actuation.CARTESIAN_VELOCITY)
    panda_env = environment.PandaEnvironment(robot_params)

    block = prop.Block()
    props = [block]

    block_pose_sensor = prop_pose_sensor.PropPoseSensor(prop=block, name="block")
    panda_env.add_extra_sensors([block_pose_sensor])

    panda_env.add_props(props)
    initialize_props = entity_initializer.prop_initializer.PropPlacer(
        props,
        position=distributions.Uniform(-0.5, 0.5),
        quaternion=rotations.UniformQuaternion())

    panda_env.add_entity_initializers([initialize_props])

    with panda_env.build_task_environment() as env:
        # Print the full action, observation and reward specification
        utils.full_spec(env)

        # Initialize the agent
        agent = Agent(env.action_spec())
        # Run the environment and agent either in headless mode or inside the GUI.
        if args.gui:
            app = utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=N, real_time=True)
