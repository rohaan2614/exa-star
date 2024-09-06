from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import (
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
)
import sys
from dataset import Dataset
from config import configclass
from util.log import LogDataProvider
from util.typing import ComparableMixin, constmethod

from loguru import logger
import numpy as np
import numpy.typing as npt
from pandas.core.frame import functools


class FitnessValue[G: Genome](ComparableMixin):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def max(cls) -> Self: ...

    @abstractmethod
    def _cmpkey(self) -> Tuple:
        """
        Redeclaration of virtual ComparableMixin::_cmpkey. This is the function used to determine sort order.
        Larger fitness values are considered better, so if you are using something like
        MSE or some other loss function, you should negate it for purposes of comparison.
        """
        ...


class Fitness[G: Genome, D: Dataset]:

    def __init__(self) -> None: ...

    @abstractmethod
    def compute(self, genome: G, dataset: D) -> FitnessValue[G]: ...


@dataclass
class FitnessConfig:
    ...


class Genome(ABC, LogDataProvider):

    def __init__(self, fitness: FitnessValue, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fitness: FitnessValue[Self] = fitness

    @abstractmethod
    @constmethod
    def clone(self) -> Self: ...

    @abstractmethod
    def __eq__(self, other) -> bool: ...

    def evaluate[D: Dataset](self, f: Fitness[Self, D], dataset: D) -> FitnessValue[Self]:
        self.fitness = f.compute(self, dataset)
        return self.fitness


class MSEValue[G: Genome](FitnessValue):

    @classmethod
    def max(cls) -> Self:
        return cls(sys.float_info.max)

    def __init__(self, mse: float) -> None:
        super().__init__()
        self.mse: float = mse

    def _cmpkey(self) -> Tuple:
        if math.isnan(self.mse):
            return (-math.inf, )
        else:
            return (-self.mse, )

    def __repr__(self) -> str: return f"MSEValue({self.mse})"


class GenomeOperator[G: Genome](ABC):

    def __init__(self, weight: float) -> None:
        # Relative weight used for computing genome operator probabilities.
        self.weight: float = weight

    def roll(self, p: float, rng: np.random.Generator) -> bool:
        """
        Returns true with proability 
        """
        assert 0 <= p <= 1.0
        return rng.random() < p

    def rolln(self, p: float, n_rolls: int, rng: np.random.Generator) -> npt.NDArray[np.bool]:
        assert 0 <= p <= 1.0
        assert n_rolls > 0
        return rng.random(n_rolls) < p


@dataclass(kw_only=True)
class GenomeOperatorConfig:
    weight: float = field(default=1.0)


class MutationOperator[G: Genome](GenomeOperator[G]):

    def __init__(self, weight: float) -> None:
        super().__init__(weight)

    @abstractmethod
    def __call__(self, genome: G, rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform a mutation on `genome`, modifying it in place: any cloning is the responsibility
        of the caller.
        If the mutation cannot be performed, returns `None`.
        """
        ...


class CrossoverOperator[G: Genome](GenomeOperator[G]):

    def __init__(self, weight: float) -> None:
        super().__init__(weight)

    @abstractmethod
    def __call__(self, parents: List[G], rng: np.random.Generator) -> Optional[G]:
        """
        Attempts to perform crossover with the supplied parents.
        Returns `None` if the crossover fails.
        """
        ...


@dataclass
class MutationOperatorConfig(GenomeOperatorConfig):
    ...


@dataclass
class CrossoverOperatorConfig(GenomeOperatorConfig):
    ...


class GenomeProvider[G: Genome]:

    def __init__(self) -> None: ...

    @abstractmethod
    def get_parents(self, rng: np.random.Generator) -> List[G]: ...

    @abstractmethod
    def get_genome(self, rng: np.random.Generator) -> G: ...


class OperatorSelector(ABC, LogDataProvider):

    @abstractmethod
    def choice[T: GenomeOperator](self, operators: Tuple[T, ...], rng: np.random.Generator) -> T:
        ...


@dataclass
class OperatorSelectorConfig:
    ...


class WeightedOperatorSelector(ABC, LogDataProvider):
    def choice[T: GenomeOperator](self, operators: Tuple[T, ...], rng: np.random.Generator) -> T:
        weights = [o.weight for o in operators]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        ichoice = rng.choice(len(operators), p=probabilities)
        return operators[ichoice]


@configclass(name="weighted_operator_selector", group="genome_factory/operator_selector",
             target=WeightedOperatorSelector)
class WeightedOperatorSelectorConfig(OperatorSelectorConfig):
    ...


class GenomeFactory[G: Genome, D: Dataset](ABC, LogDataProvider):

    def __init__(
        self,
        mutation_operators: Dict[str, MutationOperator[G]],
        crossover_operators: Dict[str, CrossoverOperator[G]],
        operator_selector: OperatorSelector,
    ) -> None:
        """
        We don't actually use the names here (yet), but they're present because of a limitation of hydra.
        """
        self.operators: Tuple[GenomeOperator[G]] = cast(
            Tuple[GenomeOperator[G]],
            tuple(
                list(mutation_operators.values()) +
                list(crossover_operators.values())
            ),
        )

        self.mutation_operators: Tuple[MutationOperator[G], ...] = tuple(
            mutation_operators.values()
        )

        self.crossover_operators: Tuple[CrossoverOperator[G], ...] = tuple(
            crossover_operators.values()
        )

        self.operator_selector: OperatorSelector = operator_selector

        assert len(self.mutation_operators), "You must specify at least one mutation operation."


    @abstractmethod
    def get_seed_genome(self, dataset: D, rng: np.random.Generator) -> G: ...

    def get_mutation(self, rng: np.random.Generator) -> MutationOperator[G]:
        return self.operator_selector.choice(self.mutation_operators, rng)

    def get_crossover(self, rng: np.random.Generator) -> CrossoverOperator[G]:
        return self.operator_selector.choice(self.crossover_operators, rng)

    def get_task(
        self, provider: GenomeProvider[G], rng: np.random.Generator,
    ) -> Callable[[np.random.Generator], Optional[G]]:
        operator: GenomeOperator[G] = self.operator_selector.choice(self.operators, rng)

        if isinstance(operator, MutationOperator):
            # Mutation
            mutation: MutationOperator[G] = cast(MutationOperator[G], operator)
            genome: G = provider.get_genome(rng)
            return lambda r: mutation(genome, r)
        else:
            # Crossover
            crossover: CrossoverOperator[G] = cast(CrossoverOperator[G], operator)
            return functools.partial(crossover, sorted(provider.get_parents(rng), key=lambda g: g.fitness))


@dataclass(kw_only=True)
class GenomeFactoryConfig:
    mutation_operators: Dict[str, MutationOperatorConfig] = field(default_factory=dict)
    crossover_operators: Dict[str, CrossoverOperatorConfig] = field(default_factory=dict)
    operator_selector: OperatorSelectorConfig = field(default_factory=WeightedOperatorSelectorConfig)
