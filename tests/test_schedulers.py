from numpy import array, inf
from unittest import TestCase
from schedulers.schedulers import ecmtc, mec


class TestSchedulers(TestCase):

    def test_mec_and_ecmtc_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 6
        # Task assignment capacities per resource.
        assignment_capacities = array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [1, 2, 3, 4, 5, 6]], dtype=object)
        # Monotonically increasing time costs.
        time_costs = array([[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Energy costs set to zero (OLAR algorithm doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to MEC algorithm.
        optimal_schedule, minimal_makespan, minimal_energy_consumption = mec(num_resources,
                                                                             num_tasks,
                                                                             assignment_capacities,
                                                                             time_costs,
                                                                             energy_costs)
        # Asserts for the MEC algorithm results.
        self.assertEqual(optimal_schedule[0], 3)
        self.assertEqual(optimal_schedule[1], 2)
        self.assertEqual(optimal_schedule[2], 1)
        self.assertEqual(minimal_makespan, 7.0)
        self.assertEqual(minimal_energy_consumption, 0.0)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(optimal_schedule[0], 3)
        self.assertEqual(optimal_schedule[1], 2)
        self.assertEqual(optimal_schedule[2], 1)
        self.assertEqual(minimal_makespan, 7.0)
        self.assertEqual(minimal_energy_consumption, 0.0)

    def test_mec_and_ecmtc_on_mc2mkp_paper_example_1(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 5
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC^2MKP algorithm doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7, 99]],
                             dtype=object)
        # Solution to MEC algorithm.
        optimal_schedule, minimal_makespan, minimal_energy_consumption = mec(num_resources,
                                                                             num_tasks,
                                                                             assignment_capacities,
                                                                             time_costs,
                                                                             energy_costs)
        # Asserts for the MEC algorithm results.
        self.assertEqual(optimal_schedule[0], 2)
        self.assertEqual(optimal_schedule[1], 3)
        self.assertEqual(optimal_schedule[2], 0)
        self.assertEqual(minimal_makespan, 0.0)
        self.assertEqual(minimal_energy_consumption, 7.5)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(optimal_schedule[0], 2)
        self.assertEqual(optimal_schedule[1], 3)
        self.assertEqual(optimal_schedule[2], 0)
        self.assertEqual(minimal_makespan, 0.0)
        self.assertEqual(minimal_energy_consumption, 7.5)

    def test_mec_and_ecmtc_on_mc2mkp_paper_example_2(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 8
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC^2MKP algorithm doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7, 99]],
                             dtype=object)
        # Solution to MEC algorithm.
        optimal_schedule, minimal_makespan, minimal_energy_consumption = mec(num_resources,
                                                                             num_tasks,
                                                                             assignment_capacities,
                                                                             time_costs,
                                                                             energy_costs)
        # Asserts for the MEC algorithm results.
        self.assertEqual(optimal_schedule[0], 1)
        self.assertEqual(optimal_schedule[1], 2)
        self.assertEqual(optimal_schedule[2], 5)
        self.assertEqual(minimal_makespan, 0.0)
        self.assertEqual(minimal_energy_consumption, 11.5)
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(optimal_schedule[0], 1)
        self.assertEqual(optimal_schedule[1], 2)
        self.assertEqual(optimal_schedule[2], 5)
        self.assertEqual(minimal_makespan, 0.0)
        self.assertEqual(minimal_energy_consumption, 11.5)
