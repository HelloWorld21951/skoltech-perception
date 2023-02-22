"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import dataclasses
import numpy as np

from typing import List
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry


@dataclasses.dataclass
class Particle(object):
    _state: np.ndarray
    _weight: np.ndarray

    @property
    def X(self) -> np.ndarray:
        return self._state

    @property
    def W(self) -> float:
        return self._weight

    def set_X(self, X: np.ndarray) -> None:
        assert X.shape == (3,)
        self._state = X
        return

    def set_W(self, W: float) -> None:
        assert 0.0 <= W <= 1.0
        self._weight = W
        return


@dataclasses.dataclass
class ParticlesList(object):
    _particles: List[Particle]

    def __post_init__(self) -> None:
        self._update_states_array()
        self._update_weights_array()
        return

    def _update_states_array(self) -> None:
        self._states_array = np.array([particle.X for particle in self._particles])

    def _update_weights_array(self) -> None:
        self._weights_array = np.array([particle.W for particle in self._particles])

    @property
    def array(self) -> List[Particle]:
        return self._particles

    @property
    def size(self) -> int:
        return len(self._particles)

    @property
    def states(self) -> np.ndarray:
        return self._states_array

    @property
    def weights(self) -> np.ndarray:
        return self._weights_array

    def set_states(self, states: np.ndarray) -> None:
        assert states.shape[0] == self.size
        for i in range(self.size):
            self._particles[i].set_X(states[i])
        self._update_states_array()
        return

    def set_weights(self, weights: np.ndarray) -> None:
        assert weights.shape[0] == self.size
        for i in range(self.size):
            self._particles[i].set_W(weights[i])
        self._update_weights_array()
        return


class PF(LocalizationFilter):
    def __init__(
        self,
        initial_state: np.ndarray,
        alphas: np.ndarray,
        bearing_std: float,
        num_particles: int,
        global_localization: bool,
    ) -> None:
        super(PF, self).__init__(initial_state, alphas, bearing_std)

        self._number_of_particles = num_particles
        self._default_weight = 1 / num_particles

        self.X = np.random.multivariate_normal(
            self.mu, self.Sigma, self._number_of_particles
        )
        self.W = np.ones(self.X.shape[0]) * self._default_weight

        self._particles = ParticlesList(
            [Particle(x, w) for x, w in zip(self.X, self.W)]
        )

    def _resample(self) -> None:
        uniform_X = np.random.multivariate_normal(
            self.mu, self.Sigma, self._number_of_particles
        )

        self.X = np.zeros((self._particles.size, 3,))
        self.W = np.ones(self._particles.size) * self._default_weight

        c = np.cumsum(self._particles.weights)
        r = np.random.uniform(0, self._default_weight)
        cnt_curr_X = 0
        cnt_new_X = 0
        for m in range(self._particles.size):
            a = r + m * self._default_weight
            while a > c[cnt_curr_X]:
                cnt_curr_X += 1
                if cnt_curr_X >= self._number_of_particles:
                    break
            else:
                self.X[cnt_new_X] = self._particles.states[cnt_curr_X]
                self.W[cnt_new_X] = self._particles.weights[cnt_curr_X]
                cnt_new_X += 1

        self.W *= cnt_new_X / (self._particles.size * np.sum(self.W))

        self.X[cnt_new_X:] = uniform_X[cnt_new_X:]
        self.W[cnt_new_X:] = self._default_weight

        self._particles.set_states(self.X)
        self._particles.set_weights(self.W)
        return

    def predict(self, u: np.ndarray):
        assert u.shape == (3,)
        self._resample()

        states = np.array(
            [
                sample_from_odometry(self._particles.states[i], u, self._alphas)
                for i in range(self._particles.size)
            ]
        )
        self._particles.set_states(states)

        stats = get_gaussian_statistics(self._particles.states)

        self._state_bar.mu = stats.mu
        self._state_bar.Sigma = stats.Sigma

    def update(self, z: np.ndarray):
        assert z.shape == (2,)
        observations = [
            get_observation(self._particles.states[i], int(z[1]))[0]
            for i in range(self._particles.size)
        ]

        W = gaussian.pdf(z[0], observations, np.sqrt(self._Q))
        W /= np.sum(W)

        self._particles.set_weights(W)

        stats = get_gaussian_statistics(self._particles.states)

        self._state.mu = stats.mu
        self._state.Sigma = stats.Sigma
