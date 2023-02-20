#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Mar-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru
"""

import os

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as sample2d
from typing import Tuple

from field_map import FieldMap
from tools.objects import FilterDebugData
from tools.objects import FilterInputData
from tools.objects import SimulationData
from tools.plot import plot_field
from tools.plot import plot_observation
from tools.plot import plot_robot
from tools.task import get_observation
from tools.task import sample_from_odometry


def generate_motion(t: float, dt: float):
    """
    Generates a square motion.

    :param t: Time (in seconds) for the current time step.
    :param dt: Time increment (in seconds) between consecutive steps.

    :raises ValueError if dt > 1.0
    :return: [first rotation (rad), forward distance, second rotation (rad)]
    """

    assert dt <= 1.0

    n = t / dt
    hz = 1 / dt
    i = np.mod(n, np.floor(hz) * 5)

    if i == 0:
        u = np.array([0, dt * 100, 0])
    elif i == 1 * hz:
        u = np.array([0, dt * 100, 0])
    elif i == 2 * hz:
        u = [np.deg2rad(45), dt * 100, np.deg2rad(45)]
    elif i == 3 * hz:
        u = [0, dt * 100, 0]
    elif i == 4 * hz:
        u = [np.deg2rad(45), 0, np.deg2rad(45)]
    else:
        u = np.array([0, dt * 100, 0])

    return u


def generate_data(
    initial_pose: np.ndarray,
    num_steps: float,
    alphas: np.ndarray,
    bearing_std: float,
    dt: float,
    animate: bool = False,
    plot_pause_s: float = 0.01,
) -> SimulationData:
    """
    Generates the trajectory of the robot using square path given by `generate_motion`.

    :param initial_pose: The initial pose of the robot in the field (format: np.array([x, y, theta])).
    :param num_steps: The number of time steps to generate the path for.
    :param alphas: The noise parameters of the control actions (format: np.array([a1, a2, a3, a4])).
    :param bearing_std: The noise parameter of observations.
    :param dt: The time difference (in seconds) between two consecutive time steps.
    :param animate: If True, this function will animate the generated data in a plot.
    :param plot_pause_s: The time (in seconds) to pause the plot animation between two consecutive frames.
    :return: SimulationData object.
    """

    # Initializations

    # State format: [x, y, theta]
    state_dim = 3
    # Motion format: [drot1, dtran, drot2]
    motion_dim = 3
    # Observation format: [bearing, marker_id]
    observation_dim = 2

    if animate:
        plt.figure(1)
        plt.ion()

    data_length = num_steps + 1
    filter_data = FilterInputData(
        np.zeros((data_length, motion_dim)), np.zeros((data_length, observation_dim))
    )
    debug_data = FilterDebugData(
        np.zeros((data_length, state_dim)),
        np.zeros((data_length, state_dim)),
        np.zeros((data_length, observation_dim)),
    )

    debug_data.real_robot_path[0] = initial_pose
    debug_data.noise_free_robot_path[0] = initial_pose

    field_map = FieldMap()
    # Covariance of observation noise.
    Q = np.diag([bearing_std ** 2, 0])

    for i in range(1, data_length):
        # Simulate Motion

        # Noise-free robot motion command.
        t = i * dt
        filter_data.motion_commands[i, :] = generate_motion(t, dt)

        # Noise-free robot pose.
        debug_data.noise_free_robot_path[i, :] = sample_from_odometry(
            debug_data.noise_free_robot_path[i - 1],
            filter_data.motion_commands[i],
            np.array([0, 0, 0, 0]),
        )

        # Move the robot based on the noisy motion command execution.
        debug_data.real_robot_path[i, :] = sample_from_odometry(
            debug_data.real_robot_path[i - 1], filter_data.motion_commands[i], alphas
        )

        # Simulate Observation

        # (n / 2) causes each landmark to be viewed twice.
        lm_id = int(np.mod(np.floor(i / 2), field_map.num_landmarks))
        debug_data.noise_free_observations[i, :] = get_observation(
            debug_data.real_robot_path[i], lm_id
        )

        # Generate observation noise.
        observation_noise = sample2d(np.zeros(observation_dim), Q)

        # Generate noisy observation as observed by the robot for the filter.
        filter_data.observations[i, :] = (
            debug_data.noise_free_observations[i] + observation_noise
        )

        if animate:
            plt.clf()

            plot_field(lm_id)
            plot_robot(debug_data.real_robot_path[i])
            plot_observation(
                debug_data.real_robot_path[i],
                debug_data.noise_free_observations[i],
                filter_data.observations[i],
            )

            plt.plot(
                debug_data.real_robot_path[1:i, 0],
                debug_data.real_robot_path[1:i, 1],
                "b",
            )
            plt.plot(
                debug_data.noise_free_robot_path[1:i, 0],
                debug_data.noise_free_robot_path[1:i, 1],
                "g",
            )

            plt.draw()
            plt.pause(plot_pause_s)

    if animate:
        plt.show(block=True)

    # This only initializes the sim data with everything but the first entry (which is just the prior for the sim).
    filter_data.motion_commands = filter_data.motion_commands[1:]
    filter_data.observations = filter_data.observations[1:]
    debug_data.real_robot_path = debug_data.real_robot_path[1:]
    debug_data.noise_free_robot_path = debug_data.noise_free_robot_path[1:]
    debug_data.noise_free_observations = debug_data.noise_free_observations[1:]

    return SimulationData(num_steps, filter_data, debug_data)


def save_data(data: Tuple, file_path: str) -> None:
    """
    Saves the simulation's input data to the given filename.

    :param data: A tuple with the filter and debug data to save.
    :param file_path: The the full file path to which to save the data.
    """

    output_dir = os.path.dirname(file_path)
    print(output_dir, file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, "wb") as data_file:
        np.savez(
            data_file,
            num_steps=data.num_steps,
            noise_free_motion=data.filter.motion_commands,
            real_observations=data.filter.observations,
            noise_free_observations=data.debug.noise_free_observations,
            real_robot_path=data.debug.real_robot_path,
            noise_free_robot_path=data.debug.noise_free_robot_path,
        )


def load_data(data_filename: str) -> SimulationData:
    """
    Load existing data from a given filename.
    Accepted file formats are pickled `npy` and MATLAB `mat` extensions.

    :param data_filename: The path to the file with the pre-generated data.
    :raises Exception if the file does not exist.
    :return: DataFile type.
    """

    if not os.path.isfile(data_filename):
        raise Exception("The data file {} does not exist".format(data_filename))

    file_extension = data_filename[-3:]
    if file_extension not in {"mat", "npy"}:
        raise TypeError(
            "{} is an unrecognized file extension. Accepted file "
            'formats include "npy" and "mat"'.format(file_extension)
        )

    num_steps = 0
    filter_data = None
    debug_data = None

    if file_extension == "npy":
        with np.load(data_filename) as data:
            # num_steps = np.asscalar(data["num_steps"])
            num_steps = np.ndarray.item(data["num_steps"])
            filter_data = FilterInputData(
                data["noise_free_motion"], data["real_observations"]
            )
            debug_data = FilterDebugData(
                data["real_robot_path"],
                data["noise_free_robot_path"],
                data["noise_free_observations"],
            )
    elif file_extension == "mat":
        data = scipy.io.loadmat(data_filename)
        if "data" not in data:
            raise TypeError("Unrecognized data file")

        data = data["data"]
        num_steps = data.shape[0]

        # Convert to zero-indexed landmark IDs.
        data[:, 1] -= 1
        data[:, 6] -= 1

        filter_data = FilterInputData(data[:, 2:5], data[:, 0:2])
        debug_data = FilterDebugData(data[:, 7:10], data[:, 10:13], data[:, 5:7])

    return SimulationData(num_steps, filter_data, debug_data)
