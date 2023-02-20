#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from tools.task import wrap_angle

ekf = {}
ekf["input"] = np.load("ekf_data/input_data.npy")
ekf["output"] = np.load("ekf_data/output_data.npy")

# pf = {}
# pf['input'] = np.load('out_pf/input_data.npy')
# pf['output'] = np.load('out_pf/output_data.npy')

filters = [ekf]

fig, axes = plt.subplots(2, 3, figsize=(13, 6), num="Filter data plots")
fig.subplots_adjust(0.05, 0.1, 0.95, 0.95, wspace=0.3, hspace=0.3)
for j, filter in enumerate(zip(filters, ["EKF"])):
    filter_name = filter[1]
    filter = filter[0]
    for i, coordinate in enumerate(["x", "y", "th"]):
        data_error = (
            filter["output"]["mean_trajectory"][:, i]
            - filter["input"]["real_robot_path"][:, i]
        )
        data_sigma = np.sqrt(filter["output"]["covariance_trajectory"][i, i, :])
        sigma_1 = (
            100
            - (sum(abs(data_error) > data_sigma) / filter["input"]["num_steps"] * 100)
        ).round(2)
        sigma_3 = (
            100
            - (
                sum(abs(data_error) > 3 * data_sigma)
                / filter["input"]["num_steps"]
                * 100
            )
        ).round(2)
        if i == 2:
            for k in range(len(data_error)):
                data_error[k] = wrap_angle(data_error[k])
        ax = axes[j, i]
        ax.fill_between(
            np.arange(filter["input"]["num_steps"]),
            -3 * data_sigma,
            3 * data_sigma,
            color="r",
            alpha=0.4,
        )
        ax.fill_between(
            np.arange(filter["input"]["num_steps"]),
            -data_sigma,
            data_sigma,
            color="g",
            alpha=0.4,
        )
        ax.plot(data_error)
        ax.set_title(
            f"{filter_name}: {coordinate} | $\sigma_1$: {sigma_1}%, $\sigma_3$: {sigma_3}%"
        )
        ax.legend(["$3\sigma$", "$\sigma$", coordinate], loc="upper left")
        ax.set_ylabel("rad" if coordinate == "th" else "cm")
        ax.set_xlabel("Step")
        ax.grid()

# plt.savefig('taskC', dpi=150)
plt.show()
