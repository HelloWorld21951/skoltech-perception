import numpy as np
import matplotlib.pyplot as plt

from tools.task import wrap_angle

filters = ["EKF", "PF"]
movement_params = ["x", "y", "theta"]

ekf_input_file = "ekf_data/input_data.npy"
ekf_output_file = "ekf_data/output_data.npy"

pf_input_file = "pf_data/input_data.npy"
pf_output_file = "pf_data/output_data.npy"

figure, axes = plt.subplots(2, 3, figsize=(15, 6), num="Filters results")
figure.subplots_adjust(0.05, 0.1, 0.95, 0.95, hspace=0.35)


def plot_filter(input_file: str, output_file: str, row: int) -> None:
    _input = np.load(input_file)
    _output = np.load(output_file)

    for i, param in enumerate(movement_params):
        error = _output["mean_trajectory"][:, i] - _input["real_robot_path"][:, i]
        if param == "theta":
            for j in range(len(error)):
                error[j] = wrap_angle(error[j])
        cov = _output["covariance_trajectory"][i, i, :]
        sigma = np.sqrt(cov)

        out_of_3_sigma = []
        err_in_3_sigma_percent = 100
        for j in range(error.shape[0]):
            if np.abs(error[j]) > 3 * sigma[j]:
                out_of_3_sigma.append(j)
                err_in_3_sigma_percent -= 100 / _input["num_steps"]

        axes[row - 1, i].fill_between(
            x=np.arange(_input["num_steps"]),
            y1=-3 * sigma,
            y2=3 * sigma,
            color="red",
            alpha=0.15,
        )
        axes[row - 1, i].plot(
            error,
            markevery=out_of_3_sigma,
            marker="o",
            mfc="red",
            mec="red",
            markersize=5,
        )

        axes[row - 1, i].set_title(
            f"{filters[row - 1]}: {param.capitalize()} error in 3*sigma: {err_in_3_sigma_percent}%"
        )
        axes[row - 1, i].set_ylabel("rad" if param == "theta" else "cm")
        axes[row - 1, i].set_xlabel("Step")

        axes[row - 1, i].grid()
    figure.show()


plot_filter(ekf_input_file, ekf_output_file, 1)
plot_filter(pf_input_file, pf_output_file, 2)
plt.show()
