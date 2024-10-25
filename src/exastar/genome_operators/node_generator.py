from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import configclass
from exastar.genome import EXAStarGenome, Node
from exastar.genome.component.dt_node import DTNode

import numpy as np


class NodeGenerator[G: EXAStarGenome](ABC):
    """
    Interface for an EXAStar node generator.
    """

    @abstractmethod
    def __call__(self, depth: float, target_genome: G, rng: np.random.Generator) -> Node:
        """
        Creates a new node for a computational graph at the
        given depth.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for a computational graph.
        """
        ...


@dataclass
class NodeGeneratorConfig:
    ...


class EXAStarNodeGenerator(NodeGenerator[EXAStarGenome]):
    """
    This is a node generator for the EXA-GP algorithm. It will
    create nodes from a selection of genetic programming operation
    nodes.

    This is the "default" node generator.
    """

    def __init__(self):
        """
        Initializes a node generator for EXA-GP.
        """
        super().__init__()

    def __call__(self, depth: float, target_genome: EXAStarGenome, rng: np.random.Generator, parameter_name: str, sign: int) -> Node:
        """
        Creates a new recurrent node for an EXA-GP computational
        graph genome. It will select from all possible node types
        uniformly at random.

        Args:
            depth: is the depth of the new node
            target_genome: is the genome the edge will be created for.

        Returns:
            A new node for an EXA-GP computational graph.
        """

        new_node = DTNode(depth, parameter_name=parameter_name, sign=sign)

        return new_node


@configclass(name="base_exagp_node_generator", group="genome_factory/node_generator",
             target=EXAStarNodeGenerator)
class EXAStarNodeGeneratorConfig(NodeGeneratorConfig):
    ...
