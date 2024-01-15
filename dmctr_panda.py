"""Minimal working example of the dm_robotics Panda model."""
import logging
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

from dm_control.composer.variation.deterministic import Sequence

np.set_printoptions(precision=3)

import csv

N = 100
Tf = 0.1 * (N - 1)

log_list = []


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

class HMatHuman:
    def __init__(self, hmat):
        self.hmat = hmat

    @property
    def R(self):
        return self.hmat[:-1, :-1]

    @property
    def p(self):
        return self.hmat[:-1, -1]


class PoseHuman:
    def __init__(self, pose):
        self.pose = pose

    @property
    def pos(self):
        return self.pose[:3]

    @property
    def quat(self):
        return self.pose[3:]


def rmat_to_omega_vector(rmat1: np.ndarray, rmat2: np.ndarray):
    A = rmat2 * rmat1.T
    theta = np.arccos((np.trace(A) - 1) * 0.5)
    W = (1 / (2 * 0.1)) * (theta / np.sin(theta)) * (A - A.T)
    omega = mr.so3ToVec(W)
    return omega


class Agent:
    def __init__(self, spec: specs.BoundedArray) -> None:
        self._spec = spec
        self.xtrajectory = None
        self.step_no = 0
        self.current_time = 0
        self.gripper_state = 1

        self.last_action = None

    def calculate_trajectory(self, timestep: dm_env.TimeStep):
        t = 0.1

        block_pose = PoseHuman(timestep.observation['block_pose'])
        tcp_pose = PoseHuman(timestep.observation['panda_tcp_pose'])
        block_hmat = tr.pos_quat_to_hmat(block_pose.pos, block_pose.quat)
        tcp_hmat = tr.pos_quat_to_hmat(tcp_pose.pos, tcp_pose.quat)

        logging.info(f"Initial block pos = {block_pose.pos} -- Initial tcp_pose = {tcp_pose.pos}")

        self.xtrajectory = mr.CartesianTrajectory(tcp_hmat, block_hmat, Tf, N + 1, 5)

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        if not self.xtrajectory:
            self.calculate_trajectory(timestep)

        action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        action[-1] = self.gripper_state
        tcp_pose_mj = timestep.observation['panda_tcp_pose']
        actual_tcp_pose = PoseHuman(tcp_pose_mj)

        if self.step_no == 0:
            action[:3] = ((HMatHuman(self.xtrajectory[0]).p - actual_tcp_pose.pos) / 0.1)
            self.step_no = 1

        elif self.step_no < N:
            expected_tcp_pos = HMatHuman(self.xtrajectory[self.step_no - 1]).p

            if np.linalg.norm(expected_tcp_pos - actual_tcp_pose.pos) < 0.01:
                logging.info(f"Step {self.step_no} completed taking step {self.step_no + 1}")
                self.step_no += 1

                action[:3] = ((HMatHuman(self.xtrajectory[self.step_no]).p - actual_tcp_pose.pos) / 0.1)
            else:
                logging.info(f"Repeating with desired point as last step {self.step_no}")
                action[:3] = ((HMatHuman(self.xtrajectory[self.step_no]).p - actual_tcp_pose.pos) / 0.1)

            self.last_action = action

            if self.step_no > 0:
                logging.info(f"Last step:{self.step_no} -- expected - {expected_tcp_pos}, "
                             f"actual = {actual_tcp_pose.pos} -- action: {action[:3]} -- "
                             f"cmd_pos: {actual_tcp_pose.pos + action[:3]* 0.1} -- "
                             f"dist= {np.linalg.norm(expected_tcp_pos - actual_tcp_pose.pos)}")

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

    block1 = prop.Block()
    props = [block1]

    block_pose_sensor = prop_pose_sensor.PropPoseSensor(prop=block1, name="block")
    panda_env.add_extra_sensors([block_pose_sensor])

    panda_env.add_props(props)
    theta = (np.pi/2) * np.random.random()
    initialize_props = entity_initializer.prop_initializer.PropPlacer(
        props,
        position=[0.4, -0.4, 0.1],
        quaternion=rotations.UniformQuaternion())

    panda_env.add_entity_initializers([initialize_props])

    with panda_env.build_task_environment() as env:
        # Print the full action, observation and reward specification
        utils.full_spec(env)

        # Initialize the agent
        agent = Agent(env.action_spec())
        # Run the environment and agent either in headless mode or inside the GUI.
        if args.gui:
            app = utils.ApplicationWithPlot(width=1920, height=1080)
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=N, real_time=True)

        # with open("log_list.csv", "w") as f:
        #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #     lines = []
        #     for line in log_list:
        #         templ = list()
        #         for l in line:
        #             for el in l:
        #                 templ.append(el)
        #         lines.append(templ)
        #     for line in lines:
        #         str_line = [str(el) for el in line]
        #         wr.writerow(str_line)
