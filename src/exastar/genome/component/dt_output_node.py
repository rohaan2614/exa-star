from typing import Optional

from exastar.genome.component.dt_node import DTNode, node_inon_t
from util.typing import overrides


class DTOutputNode(DTNode):
    def __init__(
        self,
        node_name: str,
        depth: float,
        inon: Optional[node_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ) -> None:
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
        Provides a unique string representation for this output node.
        """
        return (
            "OutputNode("
            f"parameter='{self.node_name}', "
            f"depth={self.depth}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )
