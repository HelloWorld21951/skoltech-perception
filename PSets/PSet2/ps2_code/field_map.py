"""
Sudhanva Sreesha
ssreesha@umich.edu
21-Mar-2018

Gonzalo Ferrer,
g.ferrer@skoltech.ru

Defines the field (a.k.a. map) for this task.
"""

import numpy as np


class FieldMap(object):
    def __init__(self) -> None:
        self._complete_size_x = self._inner_size_x + 2 * self._inner_offset_x
        self._complete_size_y = self._inner_size_y + 2 * self._inner_offset_y

        self._landmark_poses_x = np.array(
            [
                self._landmark_offset_x,
                self._landmark_offset_x + 0.5 * self._landmark_distance_x,
                self._landmark_offset_x + self._landmark_distance_x,
                self._landmark_offset_x + self._landmark_distance_x,
                self._landmark_offset_x + 0.5 * self._landmark_distance_x,
                self._landmark_offset_x,
            ]
        )

        self._landmark_poses_y = np.array(
            [
                self._landmark_offset_y,
                self._landmark_offset_y,
                self._landmark_offset_y,
                self._landmark_offset_y + self._landmark_distance_y,
                self._landmark_offset_y + self._landmark_distance_y,
                self._landmark_offset_y + self._landmark_distance_y,
            ]
        )

    @property
    def _inner_offset_x(self) -> int:
        return 32

    @property
    def _inner_offset_y(self) -> int:
        return 13

    @property
    def _inner_size_x(self) -> int:
        return 420

    @property
    def _inner_size_y(self) -> int:
        return 270

    @property
    def _landmark_offset_x(self) -> int:
        return 21

    @property
    def _landmark_offset_y(self) -> int:
        return 0

    @property
    def _landmark_distance_x(self) -> int:
        return 442

    @property
    def _landmark_distance_y(self) -> int:
        return 292

    @property
    def num_landmarks(self) -> int:
        return 6

    @property
    def complete_size_x(self) -> int:
        return self._complete_size_x

    @property
    def complete_size_y(self) -> int:
        return self._complete_size_y

    @property
    def landmarks_poses_x(self) -> np.ndarray:
        return self._landmark_poses_x

    @property
    def landmarks_poses_y(self) -> np.ndarray:
        return self._landmark_poses_y
