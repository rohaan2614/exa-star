from dataclasses import field
from enum import Enum
import sys
import traceback
from typing import Any, Dict, Generator, List, Optional, Self, Tuple

import dill

from config import configclass
from dataset import Dataset
from evolution.evolutionary_strategy import EvolutionaryStrategy, EvolutionaryStrategyConfig
from genome import Genome
from evolution.init_task import InitTask, InitTaskConfig


import loguru
from loguru import logger
from mpi4py import MPI
import numpy as np


class Tags(Enum):
    TASK = 0
    RESULT = 1
    FINALIZE = 2


class MPIEvolutionaryStrategy[G: Genome, D: Dataset](EvolutionaryStrategy[G, D]):

    def __init__(self, init_tasks: Dict[str, InitTask[Self]], **kwargs) -> None:
        super().__init__(**kwargs)

        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.done: bool = False

        # Switch the pickle implementation in MPI to dill, which is more flexible / can serialize more things.
        MPI.pickle.__init__(dill.dumps, dill.loads)

        # Add rank to log lines
        loguru.logger.remove()
        loguru.logger.add(
            sys.stderr,
            format="| <level>{level: <6}</level>| RANK " + str(self.rank) +
            " | <cyan>{name}.{function}</cyan>:<yellow>{line}</yellow> | {message}"
        )

        for task_name, init_task in init_tasks.items():
            logger.info(f"Executing initialization task '{task_name}' on rank {self.rank}")
            init_task.run(init_task.values(self))

    def recv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG) -> Tuple[MPI.Status, Any]:
        status: MPI.Status = MPI.Status()
        logger.info(f"Waiting for message from {source}")
        obj: Any = self.comm.recv(source=source, status=status, tag=tag)

        logger.info(f"Received {obj} from {status.Get_source()}")

        return (status, obj)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None: ...

    def abort(self, e: Exception) -> None:
        error_message = traceback.format_exc()
        logger.info(f"FAILED with exception '{e}':\n{error_message}")
        logger.info("Exiting prematurely :(...")

        # This calls sys.exit internally
        self.comm.Abort()


class AsyncMPIMasterStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_workers: int = self.comm.Get_size() - 1
        self.n_generations: int = 0

        assert self.n_workers > 0, "MPI was launched with only 1 process, there must be at least 2."

    def __enter__(self) -> Self:
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return self

    def __exit__(self, *_) -> None:
        for _ in range(self.n_workers):
            status, _ = self.recv()
            self.comm.send(None, status.Get_source(), tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {status.Get_source()}")

        logger.info("Sent finalize message to all workers...")
        super().__exit__()

    def step(self) -> None:
        status, obj = self.recv()

        if obj is not None:
            assert type(obj) is list
            print(obj)
            for genome in obj:
                assert genome is None or isinstance(genome, Genome)

            self.population.integrate_generation(obj)

        logger.info("About to generate tasks...")
        tasks = self.population.make_generation(self.genome_factory, self.rng)
        logger.info(f"Generated tasks: {tasks} for {status.Get_source()}")

        self.comm.send(tasks, dest=status.Get_source(), tag=Tags.TASK.value)

        logger.info("Sent tasks")

    def get_log_path(self) -> str:
        return f"{self.output_directory}/log.csv"

    def run(self):
        try:
            with self:
                while True:
                    logger.info(f"Starting step {self.n_generations} / {self.nsteps}")
                    self.step()

                    self.update_log(self.n_generations)
                    self.n_generations += 1

                    if self.n_generations >= self.nsteps:
                        logger.info(f"Ending on step {self.n_generations} / {self.nsteps}")
                        break

        except Exception as e:
            self.abort(e)

        self.log.to_csv(self.get_log_path())


class AsyncMPIWorkerStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Number of rows written to the data log
        self.log_rows: int = 0

    def __enter__(self) -> Self:
        logger.info("Sending initialization message to master")
        self.comm.send(None, dest=0, tag=Tags.RESULT.value)
        self.population = None  # type: ignore

        return self

    def step(self) -> None:
        logger.info("Waiting for a task")
        status, obj = self.recv(source=0)

        logger.info(f"Received tasks {type(obj)} from {status.Get_source()}")

        if obj is not None:
            results = []
            for task in obj:
                logger.info(f"Evaluating task {task}")
                genome: Optional[G] = task(self.rng)

                if genome:
                    logger.info("Evaluating fitness")
                    genome.fitness = genome.evaluate(self.fitness, self.dataset)
                    logger.info("Finished evaluation")
                results.append(genome)

            logger.info("Sending results to main")
            self.comm.send(results, dest=status.Get_source())
            logger.info("Done.")
        else:
            self.done = True

    def get_log_path(self) -> str:
        return f"{self.output_directory}/worker_{self.rank}_log.csv"

    def run(self):
        try:
            with self:
                while not self.done:
                    logger.info(f"Starting step {self.log_rows}")
                    self.step()

                    self.update_log(self.log_rows)
                    self.log_rows += 1
                    logger.info(f"Ending step {self.log_rows}")

            self.log[:self.log_rows].to_csv(self.get_log_path())
        except Exception as e:
            self.log.to_csv(self.get_log_path())
            self.abort(e)


def async_mpi_strategy_factory(**kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        return AsyncMPIMasterStrategy(**kwargs)
    else:
        return AsyncMPIWorkerStrategy(**kwargs)


@configclass(name="base_async_mpi_strategy", target=async_mpi_strategy_factory)
class AsyncMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})


class SynchronousMPIMasterStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_workers: int = self.comm.Get_size() - 1
        self.n_generations: int = 0

        assert self.n_workers > 0, "MPI was launched with only 1 process, there must be at least 2."

    def __enter__(self) -> Self:
        self.population.initialize(self.genome_factory, self.dataset, self.rng)
        return self

    def __exit__(self, *_) -> None:
        workers_closed: int = 0

        logger.info("Exiting...")
        while workers_closed < self.n_workers:
            status, _ = self.recv()
            self.comm.send(None, status.Get_source(), tag=Tags.FINALIZE.value)
            logger.info(f"Finalized worker {status.Get_source()}")

        logger.info("All workers have been finalized.")
        super().__exit__()

    def step(self) -> None:
        logger.info("About to generate tasks...")
        tasks = self.population.make_generation(self.genome_factory, self.rng)

        chunks = np.array_split(tasks, self.n_workers)  # type: ignore

        for i, chunk in enumerate(chunks):
            self.comm.send(chunk, dest=i + 1, tag=Tags.TASK.value)

        logger.info("Sent tasks")

        stats, genomes = [], []

        for ichunk in range(self.n_workers):
            logger.info(f"Waiting for chunk {ichunk}")
            status, obj = self.recv()
            logger.info("Done.")
            stats.append(status)

            if obj is not None:
                assert type(obj) is list

                for g in obj:
                    assert g is None or isinstance(g, Genome)

                genomes += obj

        self.population.integrate_generation(genomes)
        logger.info("Integrated generation")

    def get_log_path(self) -> str:
        return f"{self.output_directory}/log.csv"

    def run(self):
        try:
            with self:
                while True:
                    logger.info(f"Starting step {self.n_generations} / {self.nsteps}")
                    self.step()

                    self.update_log(self.n_generations)
                    self.n_generations += 1

                    if self.n_generations >= self.nsteps:
                        logger.info(f"Ending on step {self.n_generations} / {self.nsteps}")
                        break

        except Exception as e:
            self.log.to_csv(self.get_log_path())
            self.abort(e)


class SynchronousMPIWorkerStrategy[G: Genome, D: Dataset](MPIEvolutionaryStrategy[G, D]):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Number of rows written to the data log
        self.log_rows: int = 0

    def __enter__(self) -> Self:
        self.population = None  # type: ignore
        return self

    def step(self) -> None:
        logger.info("Waiting for a task")
        status, obj = self.recv(source=0)

        logger.info(f"Received tasks {type(obj)} from {status.Get_source()}")

        if obj is not None:
            results = []
            for task in obj:
                logger.info(f"Evaluating task {task}")
                genome: Optional[G] = task(self.rng)

                if genome:
                    logger.info("Evaluating fitness")
                    genome.fitness = genome.evaluate(self.fitness, self.dataset)
                    logger.info("Finished evaluation")
                results.append(genome)

            logger.info("Sending results to main")
            self.comm.send(results, dest=status.Get_source())
            logger.info("Done.")
        else:
            self.done = True

    def get_log_path(self) -> str:
        return f"{self.output_directory}/worker_{self.rank}_log.csv"

    def run(self):
        try:
            with self:
                while not self.done:
                    logger.info(f"Starting step {self.log_rows}")
                    self.step()

                    self.update_log(self.log_rows)
                    self.log_rows += 1
                    logger.info(f"Ending step {self.log_rows}")

            self.log[:self.log_rows].to_csv(self.get_log_path())
        except Exception as e:
            logger.info(f"FAILED with exception {e}")


def sync_mpi_strategy_factory(**kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        return SynchronousMPIMasterStrategy(**kwargs)
    else:
        return SynchronousMPIWorkerStrategy(**kwargs)


@configclass(name="base_sync_mpi_strategy", target=sync_mpi_strategy_factory)
class SynchronousMPIStrategyConfig(EvolutionaryStrategyConfig):
    init_tasks: Dict[str, InitTaskConfig] = field(default_factory=lambda: {})
