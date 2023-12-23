from numpy import array, inf
from unittest import TestCase
from schedulers.schedulers import ecmtc, ecmtc_with_accuracy, mec, mec_with_accuracy


class TestSchedulers(TestCase):

    def test_mec_on_olar_paper_example(self) -> None:
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
        self.assertEqual(3, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(1, optimal_schedule[2])
        self.assertEqual(7.0, minimal_makespan)
        self.assertEqual(0.0, minimal_energy_consumption)

    def test_ecmtc_on_olar_paper_example(self) -> None:
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
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(3, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(1, optimal_schedule[2])
        self.assertEqual(7.0, minimal_makespan)
        self.assertEqual(0.0, minimal_energy_consumption)

    def test_mec_with_accuracy_on_olar_paper_example(self) -> None:
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
        # Energy costs set to zero (OLAR algorithm doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to MEC With Accuracy algorithm.
        optimal_schedule, minimal_makespan, maximal_weighted_accuracy, minimal_energy_consumption \
            = mec_with_accuracy(num_resources,
                                num_tasks,
                                assignment_capacities,
                                time_costs,
                                accuracy_gains,
                                energy_costs)
        # Asserts for the MEC With Accuracy algorithm results.
        self.assertEqual(3, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(1, optimal_schedule[2])
        self.assertEqual(7.0, minimal_makespan)
        self.assertEqual(16.5, maximal_weighted_accuracy)
        self.assertEqual(0.0, minimal_energy_consumption)

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
        # Energy costs set to zero (OLAR algorithm doesn't consider them).
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
        self.assertEqual(3, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(1, optimal_schedule[2])
        self.assertEqual(7.0, minimal_makespan)
        self.assertEqual(16.5, maximal_weighted_accuracy)
        self.assertEqual(0.0, minimal_energy_consumption)

    def test_mec_on_mc2mkp_paper_example_1(self) -> None:
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
        self.assertEqual(2, optimal_schedule[0])
        self.assertEqual(3, optimal_schedule[1])
        self.assertEqual(0, optimal_schedule[2])
        self.assertEqual(0.0, minimal_makespan)
        self.assertEqual(7.5, minimal_energy_consumption)

    def test_ecmtc_on_mc2mkp_paper_example_1(self) -> None:
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
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(2, optimal_schedule[0])
        self.assertEqual(3, optimal_schedule[1])
        self.assertEqual(0, optimal_schedule[2])
        self.assertEqual(0.0, minimal_makespan)
        self.assertEqual(7.5, minimal_energy_consumption)

    def test_mec_on_mc2mkp_paper_example_2(self) -> None:
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
        self.assertEqual(1, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(5, optimal_schedule[2])
        self.assertEqual(0.0, minimal_makespan)
        self.assertEqual(11.5, minimal_energy_consumption)

    def test_ecmtc_on_mc2mkp_paper_example_2(self) -> None:
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
        # Solution to ECMTC algorithm.
        max_makespan = inf
        optimal_schedule, minimal_makespan, minimal_energy_consumption = ecmtc(num_resources,
                                                                               num_tasks,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               max_makespan)
        # Asserts for the ECMTC algorithm results.
        self.assertEqual(1, optimal_schedule[0])
        self.assertEqual(2, optimal_schedule[1])
        self.assertEqual(5, optimal_schedule[2])
        self.assertEqual(0.0, minimal_makespan)
        self.assertEqual(11.5, minimal_energy_consumption)
