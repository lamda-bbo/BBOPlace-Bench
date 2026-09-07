from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OPERATORS = SRC / "operators"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OPERATORS))

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import mutation as mutation_module
import sampling as sampling_module


class NoOpCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2, prob=1.0)

    def _do(self, problem, X, **kwargs):
        return X


class NoOpMutation(Mutation):
    def __init__(self):
        super().__init__(prob=1.0)

    def _do(self, problem, X, **kwargs):
        return X


class CountingPlacer:
    def __init__(self):
        self.n_evaluations = 0

    def evaluate(self, X):
        X = np.asarray(X)
        self.n_evaluations += len(X)
        fitness = np.sum(X, axis=1, dtype=float)
        overlap_rate = np.zeros(len(X), dtype=float)
        macro_pos = [{"genotype": tuple(row)} for row in X]
        return fitness, overlap_rate, macro_pos


class CountingProblem(Problem):
    def __init__(self, placer, n_var=4, upper=9):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=np.zeros(n_var, dtype=int),
            xu=np.full(n_var, upper, dtype=int),
            vtype=np.int64,
        )
        self.placer = placer

    def _evaluate(self, X, out, *args, **kwargs):
        fitness, overlap_rate, macro_pos = self.placer.evaluate(X)
        out["F"] = np.asarray(fitness)
        out["overlap_rate"] = np.asarray(overlap_rate)
        out["macro_pos"] = macro_pos


class EARegressionTest(TestCase):
    def test_dummy_mutation_initializes_and_is_a_noop(self):
        operator = mutation_module.DummyMutation(SimpleNamespace())
        X = np.arange(12).reshape(3, 4)

        result = operator._do(problem=None, X=X)

        np.testing.assert_array_equal(result, X)

    def test_swap_mutation_swaps_one_coordinate_pair_per_individual(self):
        choices = iter((np.array([0, 1]), np.array([2, 3])))

        with patch.object(
            mutation_module.np.random,
            "choice",
            side_effect=lambda *args, **kwargs: next(choices),
        ):
            operator = mutation_module.MaskGuidedOptimizationSwapMutation(
                SimpleNamespace()
            )
            X = np.array(
                [
                    [10, 11, 12, 13, 110, 111, 112, 113],
                    [20, 21, 22, 23, 120, 121, 122, 123],
                ]
            )
            original = X.copy()

            result = operator._do(problem=None, X=X)

        expected = np.array(
            [
                [11, 10, 12, 13, 111, 110, 112, 113],
                [20, 21, 23, 22, 120, 121, 123, 122],
            ]
        )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(X, original)

    def test_sampling_reuses_fitness_and_respects_evaluation_budget(self):
        population_size = 4
        sampling_repeat = 2
        max_evaluations = 20
        max_generations = (
            max_evaluations // population_size - sampling_repeat + 1
        )

        discarded_evaluations = []

        def record_func(hpwl, overlap_rate, macro_pos_all):
            discarded_evaluations.extend(np.asarray(hpwl).tolist())

        args = SimpleNamespace(
            n_sampling_repeat=sampling_repeat,
            record_func=record_func,
        )
        placer = CountingPlacer()
        problem = CountingProblem(placer)
        sampler = sampling_module.GrideGuideRandomSampling(args, placer)

        algorithm = GA(
            pop_size=population_size,
            sampling=sampler,
            crossover=NoOpCrossover(),
            mutation=NoOpMutation(),
            eliminate_duplicates=False,
        )

        minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", max_generations),
            seed=1,
            copy_algorithm=False,
            verbose=False,
        )

        initial_sampling_evaluations = population_size * sampling_repeat
        expected_pymoo_evaluations = (
            max_evaluations - initial_sampling_evaluations
        )

        self.assertEqual(
            len(discarded_evaluations),
            population_size * (sampling_repeat - 1),
        )
        self.assertEqual(
            algorithm.evaluator.n_eval,
            expected_pymoo_evaluations,
        )
        self.assertEqual(placer.n_evaluations, max_evaluations)


if __name__ == "__main__":
    main()
