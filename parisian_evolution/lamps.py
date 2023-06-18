import numpy as np
from numba import jit
from matplotlib import pyplot as plt

from utils import distance

class LampsEnvironmentConfig:

    def __init__(self, **kwargs):
        self.room_size = kwargs.get('room_size', 10)
        self.problem_size = kwargs.get('problem_size', 5)
        self.grid_scale = kwargs.get('grid_scale', 10)
        self.alpha = kwargs.get('alpha', 0.2)

    def __str__(self):
        string = (
            f"room_size = {self.room_size}, "
            f"problem_size = {self.problem_size}, "
            f"grid_scale = {self.grid_scale}, "
            f"alpha = {self.alpha}"
        )
        return string

    __repr__ = __str__

class LampsEnvironment:

    def __init__(self, config):
        self.config = config
        self._init_canvas()
        self.grid_size = int(self.config.room_size * self.config.grid_scale)
        self._init_grid(self.grid_size)
        self.del_lamps()

    def _init_grid(self, grid_size):
        self.grid = np.zeros((grid_size, grid_size))

    def _init_canvas(self):
        self.room_area = self.config.room_size * self.config.room_size
        self.single_shadow_area = self.room_area / self.config.problem_size
        self.lamp_radius = np.sqrt(self.single_shadow_area / np.pi)
        # print(f"lamp radius: {self.lamp_radius}")

    def get_lamps(self):
        if self.lamps is None:
            raise ValueError("Lamps not provided. Call set_lamps first.")
        return self.lamps

    def _generate_mask(self, r, x, y):
        y_grid, x_grid = np.ogrid[-r: r + 1, -r: r + 1]
        mask = (x_grid ** 2 + y_grid ** 2) <= r ** 2
        # print(f"mask_area = {np.sum(mask)/r/r/4}")
        # print(f"should be pi/4 = {np.pi/4}")
        extended_mask = np.zeros((self.grid_size, self.grid_size))
        x_min, y_min = max(0, x - r), max(0, y - r)
        x_max, y_max = min(self.grid_size, x + r + 1), min(self.grid_size, y + r + 1)
        extended_mask[y_min: y_max, x_min: x_max] = mask[y_min - y + r: y_max - y + r, x_min - x + r: x_max - x + r]
        return extended_mask

    def _add_lamps_to_grid(self):
        for lamp in self.lamps:
            x, y = int(lamp[0] * self.config.grid_scale), int(lamp[1] * self.config.grid_scale)
            r = int(self.lamp_radius * self.config.grid_scale)
            mask = self._generate_mask(r, x, y)
            self.grid += mask

    def set_lamps(self, lamps):
        if self.lamps is not None:
            self.del_lamps()
        self.lamps = lamps
        self.num_lamps = len(lamps)
        self._add_lamps_to_grid()

    def del_lamps(self):
        self.lamps = None
        self.num_lamps = 0
        self._init_grid(self.grid_size)

    def calc_enlightened_area(self):
        return np.sum(self.grid > 0) / self.grid_size / self.grid_size
    
    def calc_overlap_area(self):
        return np.sum(self.grid > 1) / self.grid_size / self.grid_size
    
    def calc_fitness(self):
        enlightened_area = self.calc_enlightened_area()
        overlap_area = self.calc_overlap_area()
        return enlightened_area - self.config.alpha * overlap_area
    
    def display_result(self):
        print(f"num_lamps = {self.num_lamps}\n"
              f"enlightened = {self.calc_enlightened_area():4f}, "
              f"ovrelap = {self.calc_overlap_area():4f}, "
              f"fitness = {self.calc_fitness():4f}.")

    def display(self, path=None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        plt.xlim(0, self.config.room_size)
        plt.ylim(0, self.config.room_size)
        for lamp in self.lamps:
            ax.plot(lamp[0], lamp[1], 'bx')
            circle = plt.Circle(lamp, self.lamp_radius, color='r', alpha=0.2)
            ax.add_artist(circle)
        self.display_result()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def display_grid(self, path=None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        ax.imshow(self.grid, cmap='gray')
        self.display_result()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


if __name__ == "__main__":
    config = LampsEnvironmentConfig(
        room_size=10,
        problem_size=5,
        grid_scale=200
    )
    env = LampsEnvironment(config)
    env.set_lamps(tests := [[1, 1], [2, 2], [3, 3], [4, 4], [9, 1]])
    env.display()
    env.display_grid()

    env.set_lamps(tests := [[0, 1], [2, 0], [3, 3], [9, 4], [7, 8]])
    env.display()
    env.display_grid()

    env.set_lamps(tests := [[5, 5], [5, 5]])
    env.display()
    env.display_grid()