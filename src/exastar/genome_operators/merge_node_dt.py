import math
from typing import List, Optional, Tuple

from config import configclass
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome import EXAStarGenome
from exastar.genome.component.component import Component
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from util.functional import is_not_any_type

from loguru import logger
import numpy as np


class MergeNodeDT[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        logger.trace("Performing a MergeNode mutation")

        """
        Performs merges a parent and child node into a single node.
        """

        possible_nodes: list = [
            node
            for node in filter(is_not_any_type({DTInputNode, DTOutputNode}), genome.nodes)
            if not isinstance(node.input_edge.input_node, DTInputNode)
        ]

        if len(possible_nodes) < 1:
            return None

        node = rng.choice(possible_nodes, 1, replace=False)[0]

        p_node = node.input_edge.input_node
        node.disable()

        change_edge = node.input_edge

        #Determines if left or right edge gets saved
        if rng.random() > 0.5:
            kept_out_edge = node.left_output_edge
        else:
            kept_out_edge = node.right_output_edge

        if change_edge == p_node.left_output_edge:
            p_node.left_output_edge = kept_out_edge
        else:
            p_node.right_output_edge = kept_out_edge

        return genome


@configclass(name="base_merge_dt_node_mutation", group="genome_factory/mutation_operators", target=MergeNodeDT)
class MergeNodeConfig(EXAStarMutationOperatorConfig):
    ...
