import math
from typing import Optional

from config import configclass
from exastar.genome import EXAStarGenome
from exastar.genome_operators.exastar_mutation_operator import EXAStarMutationOperator, EXAStarMutationOperatorConfig
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_set_edge import DTBaseEdge
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
import numpy as np


class DTChangeNode[G: EXAStarGenome](EXAStarMutationOperator[G]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:

        logger.trace("Performing a ChangeNode mutation")
        """
            This mutation changes the parameter of a selected middle node
        """
        nodes: List[DTNode] = []
        for node in genome.nodes:
            if not isinstance(node, DTInputNode) and not isinstance(node, DTOutputNode):
                nodes.append(node)
        if len(nodes) == 0:
            return genome
        target_node = rng.choice(nodes)
        option = target_node.parameter_name[0]
        while option == target_node.parameter_name[0]:
            option = str(rng.choice(genome.options))

        target_node.parameter_name[0] = option

        return genome






@ configclass(name="base_change_dt_node_mutation", group="genome_factory/mutation_operators", target=DTChangeNode)
class SplitEdgeConfig(EXAStarMutationOperatorConfig):
    ...