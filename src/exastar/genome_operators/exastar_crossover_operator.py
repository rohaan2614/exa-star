from dataclasses import dataclass, field
from typing import List

from genome import CrossoverOperator, CrossoverOperatorConfig
from exastar.genome import EXAStarGenome
from exastar.genome_operators.node_generator import NodeGenerator, NodeGeneratorConfig, EXAStarNodeGeneratorConfig
from exastar.genome_operators.edge_generator import (
    EdgeGenerator, EdgeGeneratorConfig, EXAStarEdgeGeneratorConfig, RecurrentEdgeGenerator, RecurrentEdgeGeneratorConfig
)

from exastar.weights import WeightGenerator, WeightGeneratorConfig

import numpy as np


class EXAStarCrossoverOperator[G: EXAStarGenome](CrossoverOperator[G]):

    def __init__(
        self,
        weight: float,
        node_generator: NodeGenerator[G],
        edge_generator: EdgeGenerator[G],
        weight_generator: WeightGenerator,
    ):
        """
        Initialies a new AddNode reproduction method.

        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """
        super().__init__(weight)
        self.node_generator: NodeGenerator[G] = node_generator
        self.edge_generator: EdgeGenerator[G] = edge_generator
        self.weight_generator: WeightGenerator = weight_generator

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """

        return self._number_parents

    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        Given the parent genomes create a child genome which is a crossover
        of the two. Keep each node and edge in both parents, and then add nodes
        and edges from either parents at their given rate.
        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Crossover only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to merge nodes (e.g.,
            there were not enough hidden nodes to merge).
        """
        sorted_parents = sorted(parents)

        # start with the best parent
        child_genome = sorted_parents[0]

        for node in child_genome.nodes:
            if isinstance(node, InputNode) or isinstance(node, OutputNode):
                # keep the inputs and outputs enabled
                continue

            if node.disabled:
                # enable the node if it is in any other parent and we
                # meet the other parent selection rate
                for other_parent in sorted_parents[1:]:
                    if node.innovation_number in other_parent.node_map.keys():
                        other_node = other_parent.node_map[node.innovation_number]

                        if not other_node.disabled:
                            # if the node is not disabled in the other parent, enable
                            # it in the child genome at the other parent selection rate
                            # if the node gets enabled, we don't need to check more
                            # parents
                            enable_node = (
                                random.uniform(0.0, 1.0)
                                < self.other_parent_selection_rate
                            )
                            if enable_node:
                                node.disabled = False
                                break

            else:
                # disable nodes randomly determined by the best parent component
                # selection rate
                node.disabled = (
                    random.uniform(0.0, 1.0) > self.best_parent_selection_rate
                )

        # iterate over the other parents and add in nodes
        # determined by the other parent selection rate

        for other_parent in sorted_parents[1:]:
            for other_node in other_parent.nodes:
                if other_node.innovation_number not in child_genome.node_map.keys():
                    # if an node is in another parent, add it given the other parent
                    # selection rate
                    include_node = (
                        random.uniform(0.0, 1.0) < self.other_parent_selection_rate
                    )
                    if include_node:
                        node_copy = copy.copy(other_node)
                        node_copy.disabled = False
                        node_copy.input_edges = []
                        node_copy.output_edges = []
                        child_genome.add_node_during_crossover(node_copy)

        # now we have added all nodes from potential parents
        # go over the other parents and add in edges for nodes
        # that were included from them (this way we already have
        # the nodes added and can do lookup to reattach things)

        for node in child_genome.nodes:
            if isinstance(node, InputNode) or isinstance(node, OutputNode):
                # inputs and outputs don't need to be connected
                continue

            # if the node is not disabled, add in all of its input and output edges
            # from all other parent genomes
            if not node.disabled:
                for other_parent in sorted_parents[1:]:
                    if node.innovation_number in other_parent.node_map.keys():
                        other_node = other_parent.node_map[node.innovation_number]

                        for other_edge in (
                            other_node.input_edges + other_node.output_edges
                        ):
                            # if there is an edge in the other parent that
                            if (
                                other_edge.innovation_number
                                not in child_genome.edge_map.keys()
                            ):
                                edge_copy = copy.copy(other_edge)
                                edge_copy.disabled = False
                                edge_copy.input_node = None
                                edge_copy.output_node = None

                                child_genome.add_edge_during_crossover(edge_copy)

        child_genome.connect_edges_during_crossover()

        for node in child_genome.nodes:
            if isinstance(node, InputNode) or isinstance(node, OutputNode):
                # inputs and outputs don't need to be connected
                continue

            if len(node.input_edges) == 0:
                print(
                    f"PRE crossover added a node to a child genome with no input edges -- node: {node}"
                )

            if len(node.output_edges) == 0:
                print(
                    f"PRE crossover added a node to a child genome with no output edges -- node: {node}"
                )

        # make sure that any node that got added to the child at least has one
        # input and one output edge, which we can connect up the same way as
        # done in the AddNode mutation.
        for node in child_genome.nodes:
            if isinstance(node, InputNode) or isinstance(node, OutputNode):
                # inputs and outputs don't need to be connected
                continue

            if len(node.input_edges) == 0:
                print(
                    f"crossover added a node to a child genome with no input edges -- node: {node}"
                )

                require_recurrent = AddNode.get_require_recurrent()
                for recurrent in [True, False]:
                    AddNode.add_input_edges(
                        target_node=node,
                        genome=child_genome,
                        recurrent=recurrent,
                        require_recurrent=require_recurrent,
                        edge_generator=self.edge_generator,
                    )

                print(f"ADDING INPUT EDGES, len now: {len(node.output_edges)}!")

            if len(node.output_edges) == 0:
                print(
                    f"crossover added a node to a child genome with no output edges -- node: {node}"
                )

                require_recurrent = AddNode.get_require_recurrent()
                for recurrent in [True, False]:
                    AddNode.add_output_edges(
                        target_node=node,
                        genome=child_genome,
                        recurrent=recurrent,
                        require_recurrent=require_recurrent,
                        edge_generator=self.edge_generator,
                    )

                print(f"ADDING OUTPUT EDGES, len now: {len(node.output_edges)}!")

        for node in child_genome.nodes:
            if isinstance(node, InputNode) or isinstance(node, OutputNode):
                # inputs and outputs don't need to be connected
                continue

            if len(node.input_edges) == 0:
                print(
                    f"POST crossover added a node to a child genome with no input edges -- node: {node}"
                )

            if len(node.output_edges) == 0:
                print(
                    f"POST crossover added a node to a child genome with no output edges -- node: {node}"
                )

        self.weight_generator(child_genome, parent_genomes=parent_genomes)

        # print("CROSSOVER CHILD GENOME:")
        # print(child_genome)

        # child_genome.plot(genome_name="child genome")
        # input("Press enter to continue...")

        return child_genome


@dataclass
class EXAStarCrossoverOperatorConfig(CrossoverOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default_factory=lambda: EXAStarNodeGeneratorConfig())
    edge_generator: EdgeGeneratorConfig = field(
        default_factory=lambda: EXAStarEdgeGeneratorConfig())
    weight_generator: WeightGeneratorConfig = field(default_factory=lambda: WeightGeneratorConfig())