"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def G(self, u: np.ndarray) -> np.ndarray:
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)

        theta = self.mu[2]

        return np.array(
            [
                [1, 0, -u[1] * np.sin(theta + u[0])],
                [0, 1, u[1] * np.cos(theta + u[0])],
                [0, 0, 1],
            ]
        )

    def V(self, u: np.ndarray) -> np.ndarray:
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)

        theta = self.mu[2]

        return np.array(
            [
                [-u[1] * np.sin(theta + u[0]), np.cos(theta + u[0]), 0],
                [u[1] * np.cos(theta + u[0]), np.sin(theta + u[0]), 0],
                [1, 0, 1],
            ]
        )

    def H(self, landmark_id: int) -> np.ndarray:
        if landmark_id not in range(6):
            raise ValueError(
                f"EKF.H(): Incorrect landmark id: {landmark_id}, must be integer in [0...5]"
            )

        x_robot = self.mu_bar[0]
        y_robot = self.mu_bar[1]

        x_landmark = self._field_map.landmarks_poses_x[landmark_id]
        y_landmark = self._field_map.landmarks_poses_y[landmark_id]

        delta_x = x_landmark - x_robot
        delta_y = y_landmark - y_robot

        return np.array(
            [
                [
                    delta_y / (delta_x ** 2 + delta_y ** 2),
                    -delta_x / (delta_x ** 2 + delta_y ** 2),
                    -1,
                ]
            ]
        )

    def predict(self, u: np.ndarray) -> None:
        assert isinstance(u, np.ndarray)
        assert u.shape == (3,)

        V = self.V(u)
        G = self.G(u)

        R = V @ get_motion_noise_covariance(u, self._alphas) @ V.T

        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T
        self._state_bar.Sigma = G @ self.Sigma @ G.T + R

    def update(self, z: np.ndarray) -> None:
        assert isinstance(z, np.ndarray)
        assert z.shape == (2,)

        landmark_id = int(z[1])

        if z[1] not in range(6):
            raise ValueError(
                f"EKF.update(): Incorrect landmark id: {landmark_id}, must be integer in [0...5]"
            )

        H = self.H(landmark_id)

        K = self.Sigma_bar @ H.T @ np.linalg.inv(H @ self.Sigma_bar @ H.T + self._Q)

        self._state.mu = self.mu_bar[np.newaxis].T + K * wrap_angle(
            z[0] - get_observation(self.mu_bar, landmark_id)[0]
        )
        self._state.Sigma = (np.eye(self.mu_bar.shape[0]) - K @ H) @ self.Sigma_bar
