import bisect
from copy import deepcopy
from dataclasses import field
from typing import Dict, List, Optional, Set, Tuple

from config import configclass
from exastar.genome.component import edge_inon_t, node_inon_t
from exastar.genome.component.dt_set_edge import DTBaseEdge
from exastar.genome.component.component import Component
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.visitor.edge_distribution_visitor import EdgeDistributionVisitor
from genome import CrossoverOperator, CrossoverOperatorConfig
from exastar.genome import EXAStarGenome
from exastar.genome_operators.node_generator import NodeGenerator, NodeGeneratorConfig
from exastar.genome_operators.edge_generator import (
    EdgeGenerator, EdgeGeneratorConfig
)
from exastar.weights import WeightGenerator, WeightGeneratorConfig
from util.functional import is_not_any_instance

from loguru import logger
import numpy as np
import torch


class EXAStarDTCrossoverOperator[G: EXAStarGenome](CrossoverOperator[G]):

    def __init__(
        self,
        weight: float,
        node_generator: NodeGenerator[G],
        edge_generator: EdgeGenerator[G],
        weight_generator: WeightGenerator,
        number_parents: int = 2,
        primary_parent_selection_p: float = 1.0,
        secondary_parent_selection_p: float = 0.5,
        line_search_step_size_min: float = -0.5,
        line_search_step_size_max: float = 1.5,
    ):
        """Initialies a new Crossover reproduction method.
        Args:
            node_generator: is used to generate a new node (perform the node type selection).
            edge_generator: is used to generate a new edge (perform the edge type selection).
            weight_generator: is used to initialize weights for newly generated nodes and edges.
        """

        super().__init__(weight)
        self.node_generator: NodeGenerator[G] = node_generator
        self.edge_generator: EdgeGenerator[G] = edge_generator
        self.weight_generator: WeightGenerator = weight_generator
        self._number_parents = number_parents
        self.primary_parent_selection_p = primary_parent_selection_p
        self.secondary_parent_selection_p = secondary_parent_selection_p
        self.line_search_step_size_min = line_search_step_size_min
        self.line_search_step_size_max = line_search_step_size_max

        assert self.line_search_step_size_min < self.line_search_step_size_max
        assert all(
            np.array([self.primary_parent_selection_p,
                      self.secondary_parent_selection_p]) <= 1.0
        )

    def roll_require_current(self, rng: np.random.Generator) -> bool:
        return rng.random() < self.require_recurrent_p

    def number_parents(self):
        """
        Returns:
            The number of parents required for this reproduction method.
        """

        return self._number_parents

    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        WORK IN PROGRESS, builds off of exastar_crossover_operator
        Phases have been adjusted and out of order to adjust for descision trees, numbers set to match those
        in original file

        3-phase crossover that can combine any number of parent genomes.

        2. Select which nodes will be included from secondary parents i.e. `parents[1:]`
        1. Start with a clone of the primary parent, `parents[0]`, and randomly disable some nodes by rolling against
           `self.primary_parent_selection_p`. (ADDED PART) Replace selected node with selection from child.
        5. Perform Lamarckian weight crossover to determine weights on new genomes.

        This should yield a new genome with all nodes and edges from the primay parent (with some nodes disabled), some
        subset of nodes from secondary parents, and some subset of edges from secondary parents.

        Args:
            parent_genomes: a list of parent genomes to create the child genome from.
                Crossover only uses the first

        Returns:
            A new genome to evaluate, None if it was not possible to merge nodes (e.g.,
            there were not enough hidden nodes to merge).
        """
        logger.trace("Performing EXAStarCrossover")

        # PHASE 0: Setting up some useful data

        # all nodes with the same innovation number are grouped togehter
        grouped_nodes: Dict[node_inon_t, List[DTNode]] = {}

        # all edges with the same inon are grouped together
        grouped_edges: Dict[edge_inon_t, List[DTBaseEdge]] = {}

        # a set of edges that appear in the parents.
        all_edges: Set[DTBaseEdge] = set()

        for parent in parents:
            for node in parent.nodes:
                grouped_nodes.setdefault(node.inon, []).append(node)
            for edge in parent.edges:
                all_edges.add(edge)
                grouped_edges.setdefault(edge.inon, []).append(edge)


        child_genome = parents[0].clone()

        # PHASE 2: randomly accept some nodes from secondary children

        # For each node that exists in secondary parent(s) but not the primary parent, include enabled it if
        # it passes at least 1 roll out of n rolls where n is the number of secondary parents that have that
        # node enabled. Edges are ignored here, so we are effectively adding orphaned nodes.
        # Note: input nodes and output nodes are implicitly ignored by only considering nodes that are not in
        # `child_genome`.
        new_nodes: List[DTNode] = []
        for nodes in filter(lambda n: n[0].inon not in child_genome.inon_to_node, grouped_nodes.values()):
            n_rolls = sum(n.enabled for n in nodes)
            if n_rolls and any(self.rolln(self.secondary_parent_selection_p, n_rolls, rng)):
                # Node deepcopy does not include any input or output edges.
                node = nodes[0]
                node_copy = deepcopy(node)
                node_copy.enable()

                # child_genome.add_node(node_copy)
                new_nodes.append(node_copy)

        if len(new_nodes) == 0:
            with torch.no_grad():
                self.weight_crossover(child_genome, grouped_nodes, grouped_edges, rng)
            return child_genome

        # PHASE 1: retain all nodes and edges from primary parent, disabling some nodes randomly

        # For each node in the primary parent, include enabled nodes that pass a roll against
        # `self.primary_parent_selection_p`. For disabled nodes, roll for each instance of that node in other
        # children and enable the node if any of them pass
        # Only consider normal nodes for this pass.
        # Disable EDGES due to node behavior
        new_edges = []
        for node in filter(is_not_any_instance({DTInputNode, DTOutputNode}), child_genome.nodes):
            if node.enabled:
                isEnabled = self.roll(self.primary_parent_selection_p, rng)
                if isEnabled:
                    replace_node = rng.choice(new_nodes)
                    if node.inon not in child_genome.inon_to_node:
                        new_edges = self.swap_nodes(child_genome, node, replace_node, rng)

        # PHASE 5: Weight initialization of new components and Lamarckian weight crossover for copied components.

        # # Do weight crossover immediately as the new weights generated below may take the weight distribution of the
        # # genome as an input.
        with torch.no_grad():
            self.weight_crossover(child_genome, grouped_nodes, grouped_edges, rng)

        # Generate new weights for new edges
        self.weight_generator(child_genome, rng, targets=new_edges)

        # Done!
        return child_genome


    def disable_node(self, node):
        node.disable()
        node.input_edge.disable()
        node.left_output_edge.disable()
        node.right_output_edge.disable()

    def swap_nodes(self, genome, node1, node2, rng):
        self.disable_node(node1)
        old_in_edge = node1.input_edge
        node1_parent = node1.input_edge.input_node
        node1_l_child = node1.left_output_edge.output_node
        node1_r_child = node1.right_output_edge.output_node

        if old_in_edge.isLeft:
            input_edge = self.edge_generator(genome, node1_parent, node2, True, rng)

        else:
            input_edge = self.edge_generator(genome, node1_parent, node2, False, rng)
        genome.add_edge(input_edge)

        l_out_edge = self.edge_generator(genome, node2, node1_l_child, True, rng)
        r_out_edge = self.edge_generator(genome, node2, node1_r_child, False, rng)
        genome.add_edge(l_out_edge)
        genome.add_edge(r_out_edge)

        node2.add_input_edge(input_edge)
        node2.add_left_edge(l_out_edge)
        node2.add_right_edge(r_out_edge)
        node2.depth = node1.depth
        genome.add_node(node2)

        return input_edge, l_out_edge, r_out_edge

    def weight_crossover(
        self,
        genome: G,
        grouped_nodes: Dict[node_inon_t, List[DTNode]],
        grouped_edges: Dict[edge_inon_t, List[DTBaseEdge]],
        rng: np.random.Generator
    ) -> None:
        # TODO: We should probably move the torch.no_grad to the caller of crossover mutation
        # for node in genome.nodes:
        #     nodes = grouped_nodes[node.inon]
        #     if len(nodes) > 1:
        #         print(list(node.parameters()))
        #         print(len(list(node.parameters())))
        #         print(len(list(list(n.parameters()) for n in nodes)))
        #         self.component_crossover(node, list(node.parameters()), list(list(n.parameters()) for n in nodes), rng)

        for edge in genome.edges:
            # Some edges are newly created and wont appear in the map
            if edge.inon not in grouped_edges:
                continue

            edges = grouped_edges[edge.inon]
            if len(edges) > 1:
                # print(len(list(edge.parameters())))
                # print(len(list(list(e.parameters()) for e in edges)))
                self.component_crossover_dt(edge, edge.weight, list(e.weight for e in edges), rng)
                # self.component_crossover(edge, list(edge.parameters()), list(list(e.parameters()) for e in edges), rng)

    def component_crossover_dt(
        self,
        component: Component,
        point: torch.Tensor,
        neighbors: List[List[torch.Tensor]],
        rng: np.random.Generator
    ) -> None:
        new_weight: torch.Tensor = self.line_search(point, torch.stack(neighbors), rng)
        component.weight.data = new_weight

    def component_crossover(
        self,
        component: Component,
        points: List[torch.Tensor],
        neighbors: List[torch.Tensor],
        rng: np.random.Generator
    ) -> None:

        new_weights: List[torch.Tensor] = [
            self.line_search(points[iw], torch.stack([neighbor[iw] for neighbor in neighbors]), rng)
            for iw in range(len(points))
        ]

        for new_weight, parameter in zip(new_weights, component.parameters()):
            parameter[:] = new_weight[:]

    def roll_line_search_step_size(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.line_search_step_size_min, self.line_search_step_size_max)

    def line_search(self, point: torch.Tensor, neighbors: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        avg_gradient = (point - neighbors).mean(0)
        return point + avg_gradient * self.roll_line_search_step_size(rng)


@configclass(name="base_exastar_dt_crossover", group="genome_factory/crossover_operators", target=EXAStarDTCrossoverOperator)
class EXAStarCrossoverOperatorConfig(CrossoverOperatorConfig):
    node_generator: NodeGeneratorConfig = field(default="${genome_factory.node_generator}")  # type: ignore
    edge_generator: EdgeGeneratorConfig = field(default="${genome_factory.edge_generator}")  # type: ignore
    weight_generator: WeightGeneratorConfig = field(default="${genome_factory.weight_generator}")  # type: ignore
