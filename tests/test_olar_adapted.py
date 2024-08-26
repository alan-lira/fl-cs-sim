from numpy import array
from unittest import TestCase
from schedulers.olar_adapted import olar_adapted


class TestOLARAdapted(TestCase):

    def test_olar_adapted_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 6
        # Task assignment capacities per resource.
        assignment_capacities = array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [0, 1, 2, 3, 4, 5, 6]], dtype=object)
        # Monotonically increasing time costs.
        time_costs = array([[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Training accuracies set to zero (OLAR paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to OLAR Adapted algorithm.
        assignment = olar_adapted(num_tasks, num_resources, time_costs, assignment_capacities)
        # Organize the final solution.
        selected_clients = [index for index, value in enumerate(list(assignment)) if value > 0]
        minimal_makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for index, value in enumerate(list(assignment)):
            if value > 0:
                makespan_i = time_costs[index][value]
                energy_i = energy_costs[index][value]
                training_accuracy_i = training_accuracies[index][value]
                if makespan_i > minimal_makespan:
                    minimal_makespan = makespan_i
                energy_consumption += energy_i
                training_accuracy += training_accuracy_i
        # print("X*: {0}".format(assignment))
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), num_resources, selected_clients))
        # print("Minimal makespan (Cₘₐₓ): {0}".format(minimal_makespan))
        # print("Energy consumption (ΣE): {0}".format(energy_consumption))
        # print("Training accuracy (ΣW): {0}".format(training_accuracy))
        # Asserts for the OLAR algorithm results.
        expected_number_selected_clients = 3
        self.assertEqual(expected_number_selected_clients, len(selected_clients))
        expected_optimal_schedule = [3, 2, 1]
        self.assertSequenceEqual(expected_optimal_schedule, list(assignment))
        expected_minimal_makespan = 7.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_energy_consumption = 0.0
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)

    def test_olar_adapted_on_mc2mkp_paper_example_1(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 5
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to OLAR Adapted algorithm.
        assignment = olar_adapted(num_tasks, num_resources, time_costs, assignment_capacities)
        # Organize the final solution.
        selected_clients = [index for index, value in enumerate(list(assignment)) if value > 0]
        minimal_makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for index, value in enumerate(list(assignment)):
            if value > 0:
                makespan_i = time_costs[index][value]
                energy_i = energy_costs[index][value]
                training_accuracy_i = training_accuracies[index][value]
                if makespan_i > minimal_makespan:
                    minimal_makespan = makespan_i
                energy_consumption += energy_i
                training_accuracy += training_accuracy_i
        # print("X*: {0}".format(assignment))
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), num_resources, selected_clients))
        # print("Minimal makespan (Cₘₐₓ): {0}".format(minimal_makespan))
        # print("Energy consumption (ΣE): {0}".format(energy_consumption))
        # print("Training accuracy (ΣW): {0}".format(training_accuracy))
        # Asserts for the OLAR algorithm results.
        expected_number_selected_clients = 1
        self.assertEqual(expected_number_selected_clients, len(selected_clients))
        expected_optimal_schedule = [5, 0, 0]
        self.assertSequenceEqual(expected_optimal_schedule, list(assignment))
        expected_minimal_makespan = 0.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_energy_consumption = 10.0
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)

    def test_olar_adapted_on_mc2mkp_paper_example_2(self) -> None:
        # Number of resources and tasks.
        num_resources = 3
        num_tasks = 8
        # Task assignment capacities per resource.
        assignment_capacities = array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=object)
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Solution to OLAR Adapted algorithm.
        assignment = olar_adapted(num_tasks, num_resources, time_costs, assignment_capacities)
        # Organize the final solution.
        selected_clients = [index for index, value in enumerate(list(assignment)) if value > 0]
        minimal_makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for index, value in enumerate(list(assignment)):
            if value > 0:
                makespan_i = time_costs[index][value]
                energy_i = energy_costs[index][value]
                training_accuracy_i = training_accuracies[index][value]
                if makespan_i > minimal_makespan:
                    minimal_makespan = makespan_i
                energy_consumption += energy_i
                training_accuracy += training_accuracy_i
        # print("X*: {0}".format(assignment))
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), num_resources, selected_clients))
        # print("Minimal makespan (Cₘₐₓ): {0}".format(minimal_makespan))
        # print("Energy consumption (ΣE): {0}".format(energy_consumption))
        # print("Training accuracy (ΣW): {0}".format(training_accuracy))
        # Asserts for the OLAR algorithm results.
        expected_number_selected_clients = 2
        self.assertEqual(expected_number_selected_clients, len(selected_clients))
        expected_optimal_schedule = [5, 3, 0]
        self.assertSequenceEqual(expected_optimal_schedule, list(assignment))
        expected_minimal_makespan = 0.0
        self.assertEqual(expected_minimal_makespan, minimal_makespan)
        expected_energy_consumption = 14.0
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)
