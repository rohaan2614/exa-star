from config import configclass
from exastar.genome.seed.seed_genome_factory import SeedGenomeFactory, SeedGenomeFactoryConfig
from exastar.genome.dt_genome import DTGenome
from exastar.time_series import TimeSeries
from exastar.weights import WeightGenerator

import numpy as np


class TrivialDTGenomeFactory(SeedGenomeFactory[DTGenome]):
    def __call__(
        self,
        generation_id: int,
        dataset: TimeSeries,
        weight_generator: WeightGenerator,
        rng: np.random.Generator,
    ) -> DTGenome:
        return DTGenome.make_trivial(
            generation_id,
            dataset.input_series_names,
            dataset.output_series_names,
            weight_generator,
            rng,
            dataset.guide,
        )


@configclass(name="base_trivial_dt_seed_genome_factory", group="genome_factory/seed_genome_factory",
             target=TrivialDTGenomeFactory)
class TrivialDTSeedGenomeFactoryConfig(SeedGenomeFactoryConfig):
    ...
