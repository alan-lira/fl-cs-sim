from numpy import array, inf
from unittest import TestCase
from schedulers.ecmtc_with_accuracy import ecmtc_with_accuracy


class TestECMTCWithAccuracy(TestCase):

    def test_ecmtc_with_accuracy_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 6
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5, 6], [1, 2], [1, 2, 3, 4, 5, 6]], dtype=object)
        # Monotonically increasing time costs.
        time_costs = array([[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Monotonically increasing accuracy gains.
        accuracy_gains = array([[0, 5, 12, 15, 22, 25, 30], [0, 4, 11, 16, 21, 22, 23], [0, 6, 7, 8, 8.5, 10, 15]],
                               dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to ECMTC With Accuracy algorithm.
        max_makespan = inf
        min_weighted_accuracy = 0
        optimal_schedule, minimal_makespan, maximal_weighted_accuracy, minimal_energy_consumption \
            = ecmtc_with_accuracy(num_resources,
                                  num_tasks,
                                  assignment_capacities,
                                  time_costs,
                                  accuracy_gains,
                                  energy_costs,
                                  max_makespan,
                                  min_weighted_accuracy)
        # Asserts for the ECMTC With Accuracy algorithm results.
        expected_optimal_schedule = [3, 2, 1]
        self.assertSequenceEqual(expected_optimal_schedule, list(optimal_schedule))
        expected_minimal_makespan = 7.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_maximal_weighted_accuracy = 16.5
        self.assertEqual(expected_maximal_weighted_accuracy, maximal_weighted_accuracy)
        expected_minimal_energy_consumption = 0.0
        self.assertEqual(expected_minimal_energy_consumption, minimal_energy_consumption)
