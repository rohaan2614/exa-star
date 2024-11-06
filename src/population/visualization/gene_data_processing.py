import numpy as np
from loguru import logger
import math
from scipy.stats import rankdata
from typing import Tuple
from population.visualization.nonlinear_reduction import get_neural_net_positions
from population.visualization.linear_reduction import reduce_using_pca, reduce_using_svd, reduce_using_mds
from scipy.spatial.distance import pdist, squareform


def _gene_list_to_numerical(gene_ids, gene_index_dict):
    """
    Takes a gene list and outputs a one-hot vector.
    Args:
        gene_ids (list[[node_inon_t, edge_inon_t]]): A list of node or edge IDs.
        gene_index_dict (dict[[node_inon_t, edge_inon_t], int]): A dict that maps node or edge id to an index.

    Returns:
        np.ndarray: A one-hot encoded vector.
    """
    arr = np.zeros(len(gene_index_dict), dtype=float)
    for gene_id in gene_ids:
        arr[gene_index_dict[gene_id]] = 1

    return arr


def convert_genes_to_numerical(genome_genes: dict):
    """
    Convert all genome node or edge lists into a numerical format for further processing.

    Args:
        genome_genes (dict[int, list[[node_inon_t, edge_inon_t]]]):
            A dict that maps a genome generation number to the list of node or edge genes.

    Returns:
        dict: A dict of all node generation numbers to numerical vectors.
    """

    # we create a set of all node IDs used here
    all_unique_ids = set()

    for array in genome_genes.values():
        all_unique_ids.update(array)

    # we get a mapping of gene to index
    gene_index_dict = {gene_id: i for i, gene_id in enumerate(all_unique_ids)}

    # convert all lists of node genes into a numerical format
    genome_genes_numerical = {
        genome_id: _gene_list_to_numerical(genes, gene_index_dict)
        for genome_id, genes in genome_genes.items()}

    # matrix where the rows are genomes, and the columns are genes
    genes_matrix = np.zeros((len(genome_genes_numerical), len(all_unique_ids)), dtype=float)
    genome_id_to_index = dict()

    for i, (genome_id, genes) in enumerate(genome_genes_numerical.items()):
        genes_matrix[i] = genes
        genome_id_to_index[genome_id] = i

    return genes_matrix, genome_id_to_index


def keys_equal(genome_id_to_index1: dict[int, int], genome_id_to_index2: dict[int, int]):
    """
    Check to see if the mappings of genome to index are equivalent.

    Args:
        genome_id_to_index1 (dict): The first dict that maps a genome ID to an index.
        genome_id_to_index2 (dict): The second dict that maps a genome ID to an index.
    """

    # get the genome IDs as an intersection of the two sets
    all_keys = set(genome_id_to_index1) & set(genome_id_to_index2)

    # iterate through every genome ID
    for gid in all_keys:

        # if any of them don't match, return false
        if genome_id_to_index1[gid] != genome_id_to_index2[gid]:
            return False
    return True


def combine_gene_mats(
        genes_mat1: np.ndarray,
        genome_id_to_index1: dict,
        genes_mat2: np.ndarray,
        genome_id_to_index2: dict):
    """
    Combine the two matrices that represent the genes of the matrices.
    """

    # check to see if the id to index dicts match
    if not keys_equal(genome_id_to_index1, genome_id_to_index2):

        # if they don't, permute the second matrix so the indices match
        permuted = np.zeros_like(genes_mat2)
        for gid, idx in genome_id_to_index2.items():
            permuted[genome_id_to_index1[gid]] = genes_mat2[idx]

        # then combine
        return np.concatenate([genes_mat1, permuted], axis=1), genome_id_to_index1
    else:
        # if the keys match, just concatenate the matrices
        return np.concatenate([genes_mat1, genes_mat2], axis=1), genome_id_to_index1


def to_color(
        val: float,
        col_low: Tuple[float|int, float|int, float|int],
        col_high: Tuple[float|int, float|int, float|int]
) -> Tuple[float, float, float]:
    if math.isnan(val):
        return (0.0, 0.0, 0.0)
    else:
        r = ((1.0 - val) * col_low[0]) + (val * col_high[0])
        g = ((1.0 - val) * col_low[1]) + (val * col_high[1])
        b = ((1.0 - val) * col_low[2]) + (val * col_high[2])
        return (r, g, b)


def dict_values_to_percentiles(fitnesses: dict):
    """
    Convert the fitnesses to a ranking between 0 and 1, preserving inequalities.

    Args:
        fitnesses (dict): The dict of genome ID to fitness.
    """

    fitnesses_real = {
        gid: fitness
        for gid, fitness in fitnesses.items()
        if not (math.isnan(fitness) or math.isinf(fitness))}

    id_to_index = {gid: idx for idx, gid in enumerate(fitnesses_real)}

    fitnesses_arr = np.zeros(len(id_to_index), dtype=float)

    for gid, idx in id_to_index.items():
        fitnesses_arr[idx] = fitnesses[gid]

    ranks = rankdata(fitnesses_arr, method='average')

    normalized_ranks = (ranks - 1) / (len(fitnesses_arr) - 1)

    percentile_dict = {gid: np.float64(1) for gid in fitnesses}
    for gid, idx in id_to_index.items():
        percentile_dict[gid] = normalized_ranks[idx]

    return percentile_dict


def reduce_genome(
        genes_matrix: np.ndarray,
        genome_id_to_index: dict,
        reduction_method: str,
        reduced_size: int,
        reduction_type: str,
        best_genome_id: int=None
):
    reduced_genes_mat = None
    if reduction_method == 'nn':
        reduced_genes_mat = get_neural_net_positions(
            genome_data=genes_matrix)

    elif reduction_method == 'mds':
        random_state = np.random.randint(0, 10**9)
        reduced_genes_mat = reduce_using_mds(
            genes_matrix=genes_matrix, reduced_size=reduced_size, random_state=random_state)

    elif reduction_method == 'pca':
        reduced_genes_mat = reduce_using_pca(
            genes_matrix=genes_matrix, reduced_size=reduced_size)

    elif reduction_method == 'svd':
        reduced_genes_mat = reduce_using_svd(
            genes_matrix=genes_matrix, reduced_size=reduced_size)

    if reduced_genes_mat is not None:

        if reduction_type == 'color':
            reduced_genes_mat -= reduced_genes_mat.min(axis=0)
            reduced_genes_mat /= reduced_genes_mat.max(axis=0)

        elif (reduction_type == 'position') and (best_genome_id is not None):

            # center it around the initial genome
            reduced_genes_mat -= reduced_genes_mat[genome_id_to_index[0]]

            # rotate it
            reduced_genes_mat = apply_rotation(
                positions_matrix=reduced_genes_mat,
                genome_id_to_index=genome_id_to_index,
                best_genome_id=best_genome_id)

        # return a mapping of genome ID to reduced genome
        return {gid: reduced_genes_mat[index] for gid, index in genome_id_to_index.items()}


def apply_rotation(positions_matrix: np.ndarray, genome_id_to_index: dict, best_genome_id):

    best_genome = positions_matrix[genome_id_to_index[best_genome_id]]

    theta = np.atan2(float(best_genome[1]), float(best_genome[0]))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rot = np.array(
        [
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]],
        dtype=positions_matrix.dtype)

    rotated_positions = np.dot(positions_matrix, rot)

    return rotated_positions


def get_all_distances(
        genes_mat: np.ndarray,
        positions: dict[int, np.ndarray],
        genome_id_to_index: dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get arrays of the genome distances and the positions distances so we can compare them.

    Args:
        genes_mat (np.ndarray): The genomes as a matrix.
        positions (dict): The mapping of genome ID to position.
        genome_id_to_index (dict): The mapping of genome ID to index in genes_mat.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The arrays of distances between rows.
    """

    n = genes_mat.shape[0]
    assert(n == len(positions))

    # create an array of positions that matches the index to the genome ID
    positions_arr = np.zeros((len(genes_mat), 2), dtype=float)
    for gid, idx in genome_id_to_index.items():
        positions_arr[idx] = positions[gid]

    # generate dist pairs
    gen_distances = pdist(genes_mat)
    pos_distances = pdist(positions_arr)

    # return both
    return gen_distances, pos_distances


def compare_distances(gen_distances, pos_distances):
    projection = (np.dot(gen_distances, pos_distances) / np.dot(pos_distances, pos_distances)) * pos_distances
    return np.linalg.norm(gen_distances - projection)


def get_expected_closest_distance(positions):

    # compute all pairwise distances
    pairwise_distances = squareform(pdist(np.array([pos for pos in positions.values()])))
    np.fill_diagonal(pairwise_distances, np.inf)
    closest_distances = [float(pairwise_distances[i].min()) for i in range(pairwise_distances.shape[0])]

    median_ = np.median(closest_distances)

    if median_ > 0:
        return median_
    else:
        return np.mean(closest_distances)

