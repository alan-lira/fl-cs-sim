from numpy import array, inf
from unittest import TestCase
from schedulers.ecmtc import ecmtc


class TestECMTC(TestCase):

    def test_ecmtc_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 6
        # Task assignment capacities per resource.
        assignment_capacities = array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [1, 2, 3, 4, 5, 6]], dtype=object)
        # Monotonically increasing time costs.
        time_costs = array([[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        expected_optimal_schedule = [3, 2, 1]
        self.assertSequenceEqual(expected_optimal_schedule, list(optimal_schedule))
        expected_minimal_makespan = 7.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_minimal_energy_consumption = 0.0
        self.assertEqual(expected_minimal_energy_consumption, minimal_energy_consumption)

    def test_ecmtc_on_mc2mkp_paper_example_1(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 5
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC^2MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7, 99]],
                             dtype=object)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        expected_optimal_schedule = [2, 3, 0]
        self.assertSequenceEqual(expected_optimal_schedule, list(optimal_schedule))
        expected_minimal_makespan = 0.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_minimal_energy_consumption = 7.5
        self.assertEqual(expected_minimal_energy_consumption, minimal_energy_consumption)

    def test_ecmtc_on_mc2mkp_paper_example_2(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 8
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC^2MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7, 99]],
                             dtype=object)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        expected_optimal_schedule = [1, 2, 5]
        self.assertSequenceEqual(expected_optimal_schedule, list(optimal_schedule))
        expected_minimal_makespan = 0.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_minimal_energy_consumption = 11.5
        self.assertEqual(expected_minimal_energy_consumption, minimal_energy_consumption)
