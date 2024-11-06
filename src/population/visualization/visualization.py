import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import os
from typing import List
import numpy as np
from loguru import logger
import json


def make_dir_if_not_exists(dir_path: str):
    """
    If a directory doesn't exist, make it.

    Args:
        dir_path (str): The directory path.
    """
    if (not os.path.exists(dir_path)) or (not os.path.isdir(dir_path)):
        os.mkdir(dir_path)

def _make_fpath(base_fname: str, cur_run_directory: str, figure_save_dir: str="figures"):
    # make the figures file directory if it doesn't already exist
    make_dir_if_not_exists(figure_save_dir)

    # get the subdirectory path we want to save to
    subdir_path = os.path.join(figure_save_dir, cur_run_directory)

    # make the subdirectory if it doesn't exist
    make_dir_if_not_exists(subdir_path)

    # create the full path, set the file type to png
    fpath = os.path.join(subdir_path, f'{base_fname}')

    return fpath


def create_and_save_figure(
        graph: nx.DiGraph,
        positions: dict,
        temporal_data: List[int],
        node_size: int|float,
        node_colors: dict,
        base_fname: str,
        cur_run_directory: str,
        legend_handles=None):
    """
    Display a graph with positions and node colors.

    Args:
        graph (nd.DiGraph): The graph of relations.
        positions (dict): The positions of the nodes.
        temporal_data (List[int]): The genomes at each timestep.
        node_size (int or float): The size of the nodes.
        node_colors (dict): The colord of the nodes.
        base_fname (str): The basic name of the file for the figure to be saved to.
        cur_run_directory (str): The directory being used for this run.
    """

    # create the layout for the graph
    pos = nx.spring_layout(graph, pos=positions, fixed=positions.keys(), seed=42)

    # convert colors to list
    colors = [node_colors[node] for node in graph.nodes]
    plt.figure(figsize=(6, 6))
    nx.draw(graph, pos, with_labels=False, node_color=colors, node_size=node_size, arrows=True)
    plt.title("Family Tree")

    if legend_handles is not None:
        plt.legend(handles=legend_handles, title="Node Categories")

    _save_figure(base_fname=base_fname, cur_run_directory=cur_run_directory)

    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    show_positions_over_time(
        temporal_data, positions, node_colors, x_limits, y_limits,
        _make_fpath(base_fname=base_fname, cur_run_directory=cur_run_directory))


def _save_figure(base_fname: str, cur_run_directory: str):
    """
    Save the figure using matplotlib.pyplot.

    Args:
        base_fname (str): The file name to save the figure to.
        cur_run_directory (str): The subdirectory within the figures directory to use.
        figure_save_dir (str): The directory in the project to save the figure in.
    """

    fpath = _make_fpath(base_fname=base_fname, cur_run_directory=cur_run_directory)
    fpath = f"{fpath}.png"

    # save it, log path
    plt.savefig(fpath)
    logger.info(f"Saved figure to {fpath}")


def save_log(log_data, base_fname: str, cur_run_directory: str):
    """
    Save the figure using matplotlib.pyplot.

    Args:
        log_data (dict): A dict with all of the log data we want to save.
        base_fname (str): The file name to save the figure to.
        cur_run_directory (str): The subdirectory within the figures directory to use.
        figure_save_dir (str): The directory in the project to save the figure in.
    """

    fpath = _make_fpath(base_fname=base_fname, cur_run_directory=cur_run_directory)

    fpath = f"{fpath}.txt"

    # save it, log path
    with open(fpath, 'w') as f:
        f.write(json.dumps(log_data))
        logger.info(f"Saved log to {fpath}")


def parse_temporal_data(temporal_data, positions):
    positions_over_time = []
    for genome_list in temporal_data:
        positions_list = []
        for genome_id in genome_list:
            positions_list.append(positions[genome_id])

        positions_arr = np.array(positions_list)
        positions_over_time.append(positions_arr)

    return positions_over_time


def show_positions_over_time(temporal_data, positions, colors, x_limits, y_limits, fpath):
    """
    Display positions over time as an animation based on temporal data and save it as a video file.

    Args:
        temporal_data (list of list of int): Each inner list contains IDs representing object positions at each time step.
        positions (dict): Maps an ID to an (x, y) position.
        colors (dict): Maps an ID to an RGB color tuple (each value in the range [0, 1]).
        x_limits (tuple): (x_min, x_max) limits for the plot.
        y_limits (tuple): (y_min, y_max) limits for the plot.
        fpath (str): File path to save the animation (without extension).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    # Create a scatter plot for each object based on the number of unique IDs in temporal_data
    scatters = [ax.plot([], [], 'o')[0] for _ in range(len(temporal_data[0]))]

    # Initialize scatter plot positions and colors
    for idx, scatter in enumerate(scatters):
        scatter.set_data([], [])
        obj_id = temporal_data[0][idx]
        if obj_id in colors:
            scatter.set_color(colors[obj_id])

    # Update scatter plot for each frame
    def animate(frame):
        if frame < len(temporal_data):
            ids_at_frame = temporal_data[frame]
            for idx, obj_id in enumerate(ids_at_frame):
                if obj_id in positions:
                    x, y = positions[obj_id]
                    scatters[idx].set_data([x], [y])  # Wrap x and y in lists
                    if obj_id in colors:
                        scatters[idx].set_color(colors[obj_id])  # Set color
        return scatters

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(temporal_data), interval=500, blit=True)

    # Save the animation
    ani.save(f"{fpath}.mp4", writer="ffmpeg", dpi=100)
