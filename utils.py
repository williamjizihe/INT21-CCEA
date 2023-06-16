import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_lamp_grid(grid_size, problem_size):
    """ Function to create a grid of one lamp
    
    Args:
        grid_size (int): Size of the grid
        problem_size (int): Size of the problem
        
    Returns:
        lamp_grid (numpy.array): Grid of one lamp
    
    """
    da = 1/ grid_size
    r2 = 1 / problem_size / np.pi
    r = np.sqrt(r2)
    N = int(r * grid_size)
    lamp_grid = np.zeros((2*N+1, 2*N+1))
    for i in range(2*N+1):
        for j in range(2*N+1):
            x = (i-N) * da
            y = (j-N) * da
            if x**2 + y**2 <= r2:
                lamp_grid[i,j] = 1
    return lamp_grid

def calculate_areas(lamps, grid_size, problem_size, lamp_grid):
    """ Function to calculate the enlightened and overlap area
    This function is faster than the original one, because it use the addition of numpy arrays, 
    which is faster than the for loop.
    
    Args:
        lamps (list): List of positions of the lamps
        grid_size (int): Size of the grid
        problem_size (int): Size of the problem
        lamp_grid (numpy.array): Grid of one lamp, given by create_lamp_grid

    Returns:
        enlightened_area (float): Enlightened area
        overlap_area (float): Overlap area
    """
    rN = lamp_grid.shape[0]
    extra = rN // 2
    # Initialize the grid
    tot_grid = np.zeros((grid_size + 2 * extra, grid_size + 2 * extra))
    
    # Illuminate the grid with each lamp
    for x, y in lamps:
        # Calculate the index of the lamp in the grid
        x_min = max(0, int(x * grid_size)-1)
        y_min = max(0, int(y * grid_size)-1)
        tot_grid[x_min:x_min+rN, y_min:y_min+rN] += lamp_grid

    # Remove the extra rows and columns
    tot_grid = tot_grid[extra:-extra, extra:-extra]
    # Count the enlightened and overlap area
    enlightened_area = np.sum(tot_grid > 0) / grid_size**2
    overlap_area = 1 / problem_size * len(lamps) - enlightened_area

    return enlightened_area, overlap_area

def original_calculate_areas(lamps, grid_size, problem_size):
    """This function calculates the enlightened and overlap area of a grid of lamps,
       but it is slower than the new one.

    Args:
        lamps (list): List of positions of the lamps
        grid_size (int): Size of the grid
        problem_size (int): Size of the problem

    Returns:
        enlightened_area (float): Enlightened area
        overlap_area (float): Overlap area
    """
    # Initialize the grid
    enlightened_grid = np.zeros((grid_size, grid_size), dtype=bool)

    # Calculate the size of each small square
    da = 1 / grid_size
    r2 = 1 / problem_size / np.pi
    r = np.sqrt(r2)
    
    # Illuminate the grid with each lamp
    for x, y in lamps:
        # Calculate the range of grid points to check
        x_min = max(0, int((x - r) / da))
        x_max = min(grid_size, int((x + r) / da))
        y_min = max(0, int((y - r) / da))
        y_max = min(grid_size, int((y + r) / da))

        # Check each grid point in the range
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if not enlightened_grid[i, j]:
                    # Calculate the distance to the lamp
                    d2 = (x - i * da)**2 + (y - j * da)**2
                    # If the grid point is within the lamp's range, illuminate it
                    enlightened_grid[i, j] = d2 <= r2

    # Count the enlightened and overlap area
    enlightened_area = np.sum(enlightened_grid) / grid_size**2
    overlap_area = np.pi * r2 * len(lamps) - enlightened_area

    return enlightened_area, overlap_area

def plot_lamps(candidate, problem_size, name = 'lamp_positions.png'):
    """ This function plots the lamps in the candidate solution and save the figure.

    Args:
        candidate (list): List of positions of the lamps
        problem_size (int): Size of the problem
        name (str): Name of the figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.axis('off')  # turn off the axis
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # plot the best candidate
    radius = np.sqrt(1 / problem_size / np.pi)
    for (x, y) in candidate: 
        circle = patches.Circle((x, y), radius, edgecolor='b', facecolor='none')
        ax.add_patch(circle)
        ax.plot(x, y, 'ro')

    plt.savefig('lamp_positions.png')