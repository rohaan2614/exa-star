from typing import Optional, cast

from exastar.genome.component.dt_edge import DTEdge, edge_inon_t
from exastar.genome.component.dt_node import DTNode


from loguru import logger
import torch


class DTBaseEdge(DTEdge):
    """
    A decision tree base edge: passes weight to note
    """

    def __init__(
        self,
        input_node: DTNode,
        output_node: DTNode,
        isLeft: bool,
        inon: Optional[edge_inon_t] = None,
        enabled: bool = True,
        active: bool = True,
        weights_initialized: bool = False,
    ) -> None:
        """
        Initializes an DTBaseEdge, which simply passes the input value to the output
        node to a given node.
        Args:
            input_node: is the input node of the edge
            output_node: is the output node of the edge

            See `exastar.genome.component.Edge` for documentation of `enabled`, `active`, and `weights_initialized`.
        """
        super().__init__(input_node, output_node, isLeft, inon, enabled, active, weights_initialized)

        # Consider this uninitialized. Cast is present because of a (possible) bug in pyright.
        self.weight: torch.nn.Parameter = cast(torch.nn.Parameter, torch.nn.Parameter(torch.ones(1)))

    def __repr__(self) -> str:
        """
        Returns a unique string representation.
        """
        return (
            "DTBaseEdge("
            f"inon={self.inon}, "
            f"input_node={self.input_node.inon}, "
            f"output_node={self.output_node.inon}, "
            f"isLeft={self.isLeft}, "
            f"enabled={self.enabled}, "
            f"active={self.active}, "
            f"weight={repr(self.weight)}"
            ")"
        )

    def reset(self):
        """
        Resets the edge gradients for the next forward pass.
        """
        pass


    def forward(self, value: torch.Tensor):
        """
        Propagates the input nodes value forward to the output node.

        Args:
            time_step: the time step the value is being fed from.
            value: the output value of the input node.
        """
        assert self.is_active()

        output_value = value * self.weight
        self.output_node.input_fired(
            value=output_value
        )
