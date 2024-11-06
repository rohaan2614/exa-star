from population.visualization.visualization import (
    create_and_save_figure, make_dir_if_not_exists, save_log, show_positions_over_time)
from population.visualization.gene_data_processing import (
    convert_genes_to_numerical, combine_gene_mats, to_color, dict_values_to_percentiles,
    reduce_genome, get_all_distances, compare_distances, get_expected_closest_distance)
from datetime import datetime
import json
import networkx as nx
import os
import numpy as np
from loguru import logger
from collections.abc import Sequence
from typing import List, Tuple, Optional
from genome import Genome
from matplotlib.patches import Patch
from config import configclass
from util.log import LogDataAggregator, LogDataAggregatorConfig, LogDataProvider
from dataclasses import field


class FamilyTreeTracker[G: Genome](LogDataAggregator):
    """
    A class used to track relations between nodes, along with other attributes.
    This saves the genome data as a series of lines in a temporary file.
    """

    def __init__(self, temp_file_dir: str, delete_temp_file: bool, **kwargs):
        super().__init__(**kwargs)

        self.positioning_method = 'mds'

        # whether you delete it at the end
        self._delete_temp_file: bool = delete_temp_file

        # create a temporary filename to store genome data in
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + str(np.random.randint(0, 1)))

        # keeps a string reference to the directory that temporary files are stored in
        self.temp_file_dir = temp_file_dir

        # stores a string  reference to the filepath of the temporary file
        self.family_tracker_file = os.path.join(
            temp_file_dir,
            f'temp_genome_data_{current_time}_{str(np.random.randint(1_000, 9_999))}')

        # make the temporary file directory if it doesn't already exist
        make_dir_if_not_exists(temp_file_dir)

        self.time_step = 0

    def _genome_to_dict(self, genome: G):
        """
        Converts the genome to a format that can be saved via json.
        This is meant to be somewhat readable.

        Returns:
            dict: A dict containing the information needed to build a family tree.
        """

        node_inons = [n.inon for n in genome.nodes]
        edge_inons = [e.inon for e in genome.edges]

        return {
            'nodes': node_inons,
            'edges': edge_inons,
            'generation_number': genome.generation_number,
            'parents': genome.parents,
            'fitness': float(genome.fitness.mse),
            'time_step': self.time_step
        }

    def track_genomes(self, genomes: List[G]):
        """
        Track a list of genomes by adding them to a temporary file.

        Args:
            genomes (List[G]): A list of genomes to track.
        """
        # make sure it is actually a list
        assert isinstance(genomes, Sequence)

        # open our temporary file
        with open(self.family_tracker_file, 'a') as f:

            encoded_genomes = [self._genome_to_dict(genome) for genome in genomes]

            # store the json data as another line that is appended to the end
            f.write(json.dumps(encoded_genomes) + '\n')

            self.time_step += 1

    def load_genomes(
            self
    ) -> Optional[
        Tuple[
            nx.DiGraph,
            dict[int, list[int]],
            dict[int, list[int]],
            dict[int, float],
            dict[int, int],
            List[List[int]]]]:
        """
        Load all genomes as a graph.

        Returns:
            graph (nx.DiGraph): A graph of genomes and their family relations.
            node_genes (dict[int, list[int]]): The dict of node genes for every genome.
            edge_genes (dict[int, list[int]]): The dict of edge genes for every genome.
            fitnesses (dict[int, float]): This dict has a fitness for every genome.
        """

        # never hurts to be sure it actually exists
        if os.path.exists(self.family_tracker_file):

            # open the file we have been saving to
            with open(self.family_tracker_file, 'r') as f:

                # create a directed graph
                graph = nx.DiGraph()

                # create the dicts
                node_genes = dict()
                edge_genes = dict()
                fitnesses = dict()
                creation_times = dict()
                temporal_data = []

                # every line in the file should have a serialized genome & parents
                for line in f:
                    # read the genome data
                    genome_data_list = json.loads(line.strip())

                    temporal_data.append([genome_data['generation_number'] for genome_data in genome_data_list])

                    for genome_data in genome_data_list:

                        nodes = genome_data['nodes']
                        edges = genome_data['edges']
                        genome_id = genome_data['generation_number']
                        parents = genome_data['parents']
                        fitness = genome_data['fitness']
                        time_step = genome_data['time_step']

                        # set the attributes for this node id
                        node_genes[genome_id] = nodes
                        edge_genes[genome_id] = edges
                        fitnesses[genome_id] = fitness

                        # add the node to make sure it is in the graph
                        if not graph.has_node(genome_id):
                            graph.add_node(genome_id)
                            creation_times[genome_id] = time_step

                        assert isinstance(parents, Sequence)

                        # iterate through all parents
                        for p_id in parents:

                            # don't store self-connections
                            if genome_id != p_id:

                                # add the edge now
                                graph.add_edge(p_id, genome_id)

                # check if we need to delete the file when done
                if self._delete_temp_file:
                    self.delete_temp_file()

                return graph, node_genes, edge_genes, fitnesses, creation_times, temporal_data

    def delete_temp_file(self):
        """
        Delete the temporary file when done. Also deletes the folder if this was the only one.
        """

        # delete the temporary file
        os.remove(self.family_tracker_file)

        # directory contents
        dir_contents = os.listdir(self.temp_file_dir)
        dir_contents = [file for file in dir_contents if file != '.DS_Store']

        # if it is empty except for '.DS_Store'
        if len(dir_contents) == 0:

            # remove '.DS_Store' if it is in the folder
            ds_store_path = os.path.join(self.temp_file_dir, '.DS_Store')
            if os.path.exists(ds_store_path):
                os.remove(ds_store_path)

            # remove the directory if it is empty
            os.rmdir(self.temp_file_dir)

    def perform_visualizations(self):
        """
        For performing visualizations at the end of a run.
        """

        # load the genomes from a temporary file
        graph, node_genes, edge_genes, fitnesses, creation_times, temporal_data = self.load_genomes()

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        # take the list of gene IDs and convert to a (float) vector format
        node_genes_matrix, node_genome_id_to_index = convert_genes_to_numerical(node_genes)

        # take the list of gene IDs and convert to a (float) vector format
        edge_genes_matrix, edge_genome_id_to_index = convert_genes_to_numerical(edge_genes)

        # combine the genes matrix, so that both node and edge genes are encoded
        genes_mat, genome_id_to_index = combine_gene_mats(
            genes_mat1=node_genes_matrix,
            genome_id_to_index1=node_genome_id_to_index,
            genes_mat2=edge_genes_matrix,
            genome_id_to_index2=edge_genome_id_to_index)

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        # find the best fitness, so we can mark the best genome
        best_fitness = float('inf')
        best_genome_id = -1
        for genome_id, fitness in fitnesses.items():
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome_id = genome_id

        # get the positions by mapping to 2D
        nn_positions = reduce_genome(
            genes_matrix=genes_mat,
            genome_id_to_index=genome_id_to_index,
            reduction_method='nn',
            reduced_size=2,
            reduction_type='position',
            best_genome_id=best_genome_id)

        gen_distances, nn_distances = get_all_distances(
            genes_mat=genes_mat, positions=nn_positions, genome_id_to_index=genome_id_to_index)

        print(f"nn correlation: {np.corrcoef(gen_distances, nn_distances)[0, 1]}")
        print(f"scaled distances: {compare_distances(gen_distances, nn_distances)}")
        print()

        mds_positions = reduce_genome(
            genes_matrix=genes_mat,
            genome_id_to_index=genome_id_to_index,
            reduction_method='mds',
            reduced_size=2,
            reduction_type='position',
            best_genome_id=best_genome_id)

        gen_distances, mds_distances = get_all_distances(
            genes_mat=genes_mat, positions=mds_positions, genome_id_to_index=genome_id_to_index)

        print(f"mds correlation: {np.corrcoef(gen_distances, mds_distances)[0, 1]}")
        print(f"scaled distances: {compare_distances(gen_distances, mds_distances)}")
        print()

        positions = nn_positions

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        expected_closest_distance = get_expected_closest_distance(positions)

        print(f"expected_closest_distance: {expected_closest_distance}")

        # the size of the nodes in the visualization scale with the distances between them
        node_size = np.median(expected_closest_distance) * 500

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        # use PCA to determine colors
        pca_colors = reduce_genome(
            genes_matrix=genes_mat,
            genome_id_to_index=genome_id_to_index,
            reduction_method='pca',
            reduced_size=3,
            reduction_type='color',
            best_genome_id=best_genome_id)

        # use timings to determine colors
        timing_colors = {gid: (t / self.time_step) for gid, t in creation_times.items()}
        timing_colors = {gid: (b, b, b) for gid, b in timing_colors.items()}
        timing_colors[best_genome_id] = (1, 0, 0)

        # use fitness to determine colors
        red = (1.0, 0.0, 0.0)
        blue = (0.0, 0.0, 1.0)
        col_low = red
        col_high = blue
        fitness_colors = dict_values_to_percentiles(fitnesses)
        fitness_colors = {gid: to_color(f, col_low=col_low, col_high=col_high) for gid, f in fitness_colors.items()}

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        # set the subdirectory name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cur_run_directory = f"run_results_{current_time}"

        # create the figures...

        legend_handles = [
            Patch(color=col_low, label='Low Loss'),
            Patch(color=col_high, label='High Loss'),
        ]

        # perform the visualizations and save the resulting figures
        create_and_save_figure(
            graph, positions, temporal_data, node_size, pca_colors,
            "pca_colors", cur_run_directory)
        create_and_save_figure(
            graph, positions, temporal_data, node_size, timing_colors,
            "timing_colors", cur_run_directory)
        create_and_save_figure(
            graph, positions, temporal_data, node_size, fitness_colors,
            "fitness_colors", cur_run_directory, legend_handles)

        # ________ ________ ________ ________ ________ ________ ________ ________ ________

        log_data = dict()
        log_data["positioning_method"] = self.positioning_method
        log_data["n_timesteps"] = self.time_step - 1
        log_data["total_population_size"] = graph.number_of_nodes()
        log_data["n_population_connections"] = graph.number_of_edges()
        log_data["median_closest_dist"] = float(expected_closest_distance)
        log_data["node_size"] = float(node_size)
        save_log(log_data, base_fname="run_log", cur_run_directory=cur_run_directory)


@configclass(name="base_family_tree_tracker", group="family_tree_tracker", target=FamilyTreeTracker)
class FamilyTreeTrackerConfig(LogDataAggregatorConfig):
    temp_file_dir: str = field(default='temp_genome_data')
    delete_temp_file: bool = field(default=True)
