#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
28-February-2021
"""
import numpy as np
import mrob
import copy

from matplotlib import pyplot as plt
from tqdm import tqdm

from tools.objects import Gaussian
from tools.plot import (
    get_plots_figure,
    plot_robot,
    plot2dcov,
    plot_field,
    plot_observations,
    plot_chi2,
    plot_matrix,
)
from field_map import FieldMap
from slam.sam import Sam

from tools.data import generate_data, load_data
from tools.task import get_dummy_context_mgr, get_movie_writer
from tools.helpers import get_cli_args, validate_cli_args


def main():
    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas) ** 2
    beta = np.array(args.beta)
    beta[1] = np.deg2rad(beta[1])

    mean_prior = np.array([180.0, 50.0, 0.0])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    Q = np.diag(beta**2)

    chi2 = []

    slam = Sam(
        initial_state=initial_state,
        alphas=alphas,
        slam_type=args.filter_name,
        data_association=args.data_association,
        update_type=args.update_type,
        Q=Q,
    )

    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_data(
            initial_state.mu.T,
            args.num_steps,
            args.num_landmarks_per_side,
            args.max_obs_per_time_step,
            alphas,
            beta,
            args.dt,
        )
    else:
        raise RuntimeError("")

    should_show_plots = True if args.animate else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False

    field_map = FieldMap(args.num_landmarks_per_side)

    fig = get_plots_figure(should_show_plots, should_write_movie)
    movie_writer = get_movie_writer(
        should_write_movie, "Simulation SLAM", args.movie_fps, args.plot_pause_len
    )

    with movie_writer.saving(
        fig, args.movie_file, data.num_steps
    ) if should_write_movie else get_dummy_context_mgr():
        for t in tqdm(range(data.num_steps)):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]

            slam.predict(u)

            slam.update(z)
            slam.solve()  # Comment this out for tasks 1C and 2E

            if not should_update_plots:
                continue

            plt.cla()
            plot_field(field_map, z)
            plot_robot(data.debug.real_robot_path[t])
            plot_observations(
                data.debug.real_robot_path[t],
                data.debug.noise_free_observations[t],
                data.filter.observations[t],
            )

            plt.plot(
                data.debug.real_robot_path[1:tp1, 0],
                data.debug.real_robot_path[1:tp1, 1],
                "m",
            )
            plt.plot(
                data.debug.noise_free_robot_path[1:tp1, 0],
                data.debug.noise_free_robot_path[1:tp1, 1],
                "g",
            )

            plt.plot(
                [data.debug.real_robot_path[t, 0]],
                [data.debug.real_robot_path[t, 1]],
                "*r",
            )
            plt.plot(
                [data.debug.noise_free_robot_path[t, 0]],
                [data.debug.noise_free_robot_path[t, 1]],
                "*g",
            )

            nodes = np.array(
                [i.flatten() for i in slam.graph.get_estimated_state()], dtype=object
            )

            obs_states = list(slam.landmark_ids.values())
            non_obs_states = [i for i in range(len(nodes)) if i not in obs_states]

            nodes_for_plot_obs_states = [nodes[i] for i in obs_states]
            nodes_for_plot_non_obs_states = [nodes[i] for i in non_obs_states]

            obs_since_last_state = (len(nodes) - 1) - non_obs_states[-1]

            prev_node = nodes[slam.prev_node]
            inf_matrix = slam.graph.get_information_matrix()

            if should_show_plots:
                plt.plot(
                    [i[0] for i in nodes_for_plot_non_obs_states],
                    [i[1] for i in nodes_for_plot_non_obs_states],
                    "blue",
                )

                plt.scatter(
                    [i[0] for i in nodes_for_plot_obs_states],
                    [i[1] for i in nodes_for_plot_obs_states],
                    color="orange",
                )
                # comment this out if solve() is not called iteratively (tasks 1C and 2E)
                plot2dcov(
                    np.array(prev_node[:-1]),
                    np.linalg.inv(inf_matrix.toarray())[
                        -3 - 2 * obs_since_last_state : -1 - 2 * obs_since_last_state,
                        -3 - 2 * obs_since_last_state : -1 - 2 * obs_since_last_state,
                    ],
                    "b",
                    nSigma=3,
                )
                plt.draw()
                plt.pause(args.plot_pause_len)

            if should_write_movie:
                movie_writer.grab_frame()

    for idx in non_obs_states:
        print(
            f"Node ID: {idx}\t->\tPosition: {slam.graph.get_estimated_state()[idx].T}"
        )

    for idx in obs_states:
        print(
            f"Landmark ID: {idx}\t->\tPosition: {slam.graph.get_estimated_state()[idx].T}"
        )

    print(f"Information matrix:\n{slam.Q}")

    # slam.graph.solve(mrob.LM)  # Uncomment this for task 2E

    plot_chi2(slam.chi2, args.dt)
    plot_matrix(slam.graph.get_adjacency_matrix(), "Adjacency matrix")
    plot_matrix(slam.graph.get_information_matrix(), "Information matrix")
    plt.show(block=True)


if __name__ == "__main__":
    main()
