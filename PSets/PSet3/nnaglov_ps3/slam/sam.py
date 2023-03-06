"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian


class Sam(SlamBase):
    def __init__(
        self,
        initial_state,
        alphas,
        state_dim=3,
        obs_dim=2,
        landmark_dim=2,
        action_dim=3,
        *args,
        **kwargs,
    ):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas

        self.graph = mrob.FGraph()

        self.landmark_ids = {}
        self.chi2 = []

        self.prev_node = self.graph.add_node_pose_2d(initial_state.mu)
        self.graph.add_factor_1pose_2d(
            initial_state.mu, self.prev_node, np.linalg.inv(initial_state.Sigma)
        )
        # self.graph.print(True)

    def predict(self, u):
        # print(f"\nEstimated state:\n{self.graph.get_estimated_state()}\n")
        J_u = state_jacobian(
            self.graph.get_estimated_state()[self.prev_node].flatten(), u
        )[-1]
        new_node = self.graph.add_node_pose_2d(np.zeros(3))
        self.graph.add_factor_2poses_2d_odom(
            u,
            self.prev_node,
            new_node,
            np.linalg.inv(J_u @ get_motion_noise_covariance(u, self.alphas) @ J_u.T),
        )
        self.prev_node = new_node

    def update(self, z):
        for observation in z:
            obs_lm_id = int(observation[-1])
            initializeLandmark = not bool(self.landmark_ids.get(obs_lm_id, 0))
            if initializeLandmark:
                self.landmark_ids[obs_lm_id] = self.graph.add_node_landmark_2d(
                    np.zeros(2)
                )
            self.graph.add_factor_1pose_1landmark_2d(
                observation[:-1],
                self.prev_node,
                self.landmark_ids[obs_lm_id],
                np.linalg.inv(self.Q),
                initializeLandmark,
            )

    def solve(self):
        self.graph.solve()
        self.chi2.append(self.graph.chi2())
