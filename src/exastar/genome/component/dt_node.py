from __future__ import annotations
import bisect
import copy
from typing import List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from exastar.genome.component.dt_set_edge import DTBaseEdge

from exastar.genome.component.node import Node
from exastar.inon import inon_t
from exastar.genome.component.component import Component
from util.typing import ComparableMixin, overrides

from loguru import logger
import numpy as np
import torch


class node_inon_t(inon_t):
    """
    Define a new type so comparisons to base `inon_t` and any other child classes will return `False`.
    """
    ...


class DTNode(ComparableMixin, Component):
    """
    Neural network node for recurrent neural networks. Sortable courtesy of `ComparableMixin`, state, activation state,
    and `torch.nn.Module` stuff courtesy of `Component`.
    """

    def __init__(
            self,
            depth: float,
            parameter_name: str = None,
            sign: int = None,
            inon: Optional[node_inon_t] = None,
            enabled: bool = True,
            active: bool = True,
            weights_initialized: bool = False,
    ) -> None:
        """
        Initializes an abstract Node object with base functionality for building
        computational graphs.

        Args:
            depth: A number between 0 (input node) and 1 (output node) which represents how deep this node is within
              the computational graph.
            max_sequence_length: Is the maximum length of any time series to be processed by the neural network this
              node is part of.
            inon: Is the node's unique innovation number.

            See `exastar.genome.component.Component` for details on further arguments.
        """
        #CHECK HERE IF SOMETHING WRONG
        super().__init__(type=DTNode, enabled=enabled, active=active, weights_initialized=weights_initialized)

        self.enabled = enabled
        self.isActive = active
        self.inon: node_inon_t = inon if inon is not None else node_inon_t()
        self.depth: float = depth
        self.required_inputs: int = 0
        self.parameter_name = parameter_name
        self.sign = sign
        self.inputs_fired = 0

        self.input_edge: DTBaseEdge = None
        self.left_output_edge: DTBaseEdge = None
        self.right_output_edge: DTBaseEdge = None
        self.value: torch.Tensor = torch.zeros(1)


    def _create_value(self) -> List[torch.Tensor]:
        """
        A series of 0s for the empty state of `self.value`.
        """
        # return [torch.zeros(1) for _ in range(self.max_sequence_length)]
        return torch.zeros(1)

    def __getstate__(self):
        """
        Overrides the default implementation of object.__getstate__ because we are unable to pickle
        large networks if we include input and outptut edges. Instead, we will rely on the construction of
        new edges to add the appropriate input and output edges. See exastar.genome.component.Edge.__setstate__
        to see how this is done.

        `self.value` is not copied, meaining resumable training will not work.

        Returns:
            state dictionary sans the input and output edges
        """
        state: dict = dict(self.__dict__)

        # state["input_output_edge"] = []
        # state["right_output_edge"] = []
        # state["value"] = []

        return state

    def __setstate__(self, state):
        """
        Note that this __setstate__ will mess up training after a clone since it re-creates `self.value`.
        """
        super().__setstate__(state)
        self.value = self._create_value()

    def __deepcopy__(self, memo):
        """
        Same story as __getstate__: we want to avoid stack overflow when copying, so we exclude edges.
        """

        cls = self.__class__
        clone = cls.__new__(cls)

        memo[id(self)] = clone

        # __getstate__ defines input_edges and output_edges to be empty lists
        state = self.__getstate__()
        for k, v in state.items():
            setattr(clone, k, copy.deepcopy(v, memo))

        return clone

    @overrides(torch.nn.Module)
    def __repr__(self) -> str:
        """
        Overrides the torch.nn.Module __repr__, which prints a ton of torch information.
        This can still be accessed by calling
        ```
        node: Node = ...
        torch_repr: str = torch.nn.Module.__repr__(node)
        ```
        """
        return (
            "Node("
            f"depth={self.depth}, "
            f"depth={self.parameter_name}, "
            f"depth={self.sign}, "
            f"inon={self.inon}, "
            f"enabled={self.enabled})"
        )

    @overrides(ComparableMixin)
    def _cmpkey(self) -> Tuple:
        """
        Sort nodes by their depth.
        """
        return (self.depth,)

    @overrides(object)
    def __hash__(self) -> int:
        """
        The innovation number provides a perfect hash.
        """
        return int(self.inon)

    @overrides(object)
    def __eq__(self, other: object) -> bool:
        """
        Nodes with the same innovation number should be equal.
        """
        return isinstance(other, DTNode) and self.inon == other.inon

    def add_input_edge(self, edge: DTBaseEdge):
        """
        Adds an input edge to this node.

        Args:
            edge: a new input edge for this node.
        """
        assert edge.output_node.inon == self.inon
        self.input_edge = edge

    def add_right_edge(self, edge: DTBaseEdge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """
        assert edge.input_node.inon == self.inon
        assert not edge.inon == self.left_output_edge

        self.right_output_edge = edge

    def add_left_edge(self, edge: DTBaseEdge):
        """
        Adds an output edge to this node.

        Args:
            edge: a new output edge for this node.
        """

        assert edge.input_node.inon == self.inon
        assert not edge.inon == self.right_output_edge

        self.left_output_edge = edge


    def input_fired(self, value: torch.Tensor):
        self.value = value
        self.inputs_fired = 1

    # def input_fired(self, time_step: int, value: torch.Tensor):
    #     """
    #     Used to track how many input edges have had forward called and passed their value to this Node.
    #
    #     Args:
    #         time_step: The time step the input is being fired from.
    #         value: The tensor being passed forward from the input edge.
    #     """
    #
    #     if time_step < self.max_sequence_length:
    #         self.inputs_fired[time_step] += 1
    #
    #         # accumulate the values so we can later use them when forward
    #         # is called on the Node.
    #         self.accumulate(time_step=time_step, value=value)
    #
    #         assert self.inputs_fired[time_step] <= self.required_inputs, (
    #             f"node inputs fired {self.inputs_fired[time_step]} > len(self.input_edges): {self.required_inputs}\n"
    #             f"node {type(self)}, inon: {self.inon} at "
    #             f"depth: {self.depth}\n"
    #             f"edges:\n {self.input_edges}\n"
    #             "this should never happen, for any forward pass a node should get at most N input fireds"
    #             ", which should not exceed the number of input edges."
    #         )
    #
    #     return None

    def reset(self):
        """
        Resets the parameters and values of this node for the next forward and backward pass.
        """
        self.inputs_fired = 0
        self.value = torch.zeros(1)

    # def accumulate(self, time_step: int, value: torch.Tensor):
    #     """
    #     Used by child classes to accumulate inputs being fired to the Node. Input nodes simply act with the identity
    #     activation function so simply sum up the values.
    #
    #     Args:
    #         time_step: The time step the input is being fired from.
    #         value: The tensor being passed forward from the input edge.
    #     """
    #     self.value[time_step] = self.value[time_step] + value

    def get_parameter(self):
        return self.parameter_name

    def forward(self, parameter_val: float):
        """
        Propagates an input node's value forward for a given time step. Will check to see if any recurrent into the
        input node have been fired first.
        """
        if self.inputs_fired != 0:
            if self.sign == 0: # v < parameter
                if self.value < parameter_val:
                    if self.left_output_edge.enable:
                        self.left_output_edge.forward(value=torch.ones_like(self.value))
                        # self.right_output_edge.active = False
                else:
                    if self.right_output_edge.enable:
                        self.right_output_edge.forward(value=torch.ones_like(self.value))
                        # self.left_output_edge.active = False
            else: # v > parameter
                if self.value > parameter_val:
                    if self.left_output_edge.enable:
                        self.left_output_edge.forward(value=torch.ones_like(self.value))
                        # self.right_output_edge.active = False
                else:
                    if self.right_output_edge.enable:
                        self.right_output_edge.forward(value=torch.ones_like(self.value))
                        # self.left_output_edge.active = False
