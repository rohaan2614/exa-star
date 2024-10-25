from typing import Optional
import bisect
from exastar.genome.component.dt_node import DTNode, node_inon_t
from exastar.genome.component.dt_set_edge import DTBaseEdge
from util.typing import overrides
import torch

class DTInputNode(DTNode):
    """
    An input node is like a regular node, except it has an input parameter name.
    """

    def __init__(
        self,
        node_name: str,
        depth: float,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ):
        """
        Creates an input node of a computational graph.

        Args:
            node_name: is the input parameter name.

            See `exastar.genome.component.Node` for details on `depth`, `max_sequence_length`, and `inon`.
            See `exastar.genome.component.Component` for details on `enabled`, `active`, and `weights_initialized`.
        """
        super().__init__(depth, None, None, inon, enabled, active, weights_initialized)

        self.node_name: str = node_name

    @overrides(DTNode)
    def __repr__(self) -> str:
        """
        Provides a unique string representation for this input node.
        """
        return (
            "InputNode("
            f"parameter='{self.node_name}', "
            f"depth={self.depth}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )

    @overrides(DTNode)
    def forward(self):
        """
        Propagates an input node's value forward, only heads left at start.
        """
        if self.left_output_edge.enabled:
            self.left_output_edge.forward(value=torch.ones(1))


