from numpy import array
from numpy.random import uniform, seed
from unittest import TestCase
from schedulers.elastic_adapted import elastic_adapted


class TestELASTICAdapted(TestCase):

    def test_elastic_adapted(self) -> None:

        # Execute the adapted version of ELASTIC algorithm.
        # ELASTIC: Energy and Latency-Aware Resource Management and Client Selection for Federated Learning.
        # Goal: Optimize the tradeoff between maximizing the number of selected clients and
        # minimizing the total energy consumption.

        # Seed.
        seed(7)

        # Simulation Parameters.
        # Number of resources and tasks.
        # 30 clients.
        I = 30
        # 30 tasks.
        T = 30
        # Task assignment capacities per resource.
        assignment_capacities = []
        assignment_capacities_i = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        assignment_capacities = array(assignment_capacities)
        for i in range(I):
            assignment_capacities_i.append(list(range(0, assignment_capacities[i]+1)))
        # Time costs to process tasks per resource.
        time_costs = []
        for i in range(I):
            time_costs_i = uniform(0, 10.001, T)  # tᵢ ∼ U(0, 10) seconds.
            time_costs_i = list(time_costs_i)
            time_costs_i.insert(0, 0)
            time_costs.append(time_costs_i)
        time_costs = array(time_costs)
        # Energy costs to process tasks per resource.
        energy_costs = []
        for i in range(I):
            energy_costs_i = []
            for t in range(T):
                energy_cost_i = time_costs[i][t] * uniform(0, 5.001)  # Eᵢ ∼ U(0, 5) × tᵢ Joules.
                energy_costs_i.append(energy_cost_i)
            energy_costs.append(energy_costs_i)
        energy_costs = array(energy_costs)
        # Training accuracies per tasks per resource.
        training_accuracies = []
        for i in range(I):
            training_accuracies_i = []
            for t in range(T):
                training_accuracy_i = uniform(0, 0.015)  # εᵢ ∼ U(0, 0.15).
                training_accuracies_i.append(training_accuracy_i)
            training_accuracies.append(training_accuracies_i)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # Solution to ELASTIC's Adapted Algorithm 1: Client Selection.
        x, tasks_assignment, selected_clients \
            = elastic_adapted(I, assignment_capacities, time_costs, energy_costs, τ, α)
        makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for sel_index, num_tasks_scheduled in enumerate(list(tasks_assignment)):
            if num_tasks_scheduled > 0:
                i_index = assignment_capacities_i[sel_index].index(num_tasks_scheduled)
                time_cost_i = time_costs[sel_index][i_index]
                if time_cost_i > makespan:
                    makespan = time_cost_i
                energy_cost_i = energy_costs[sel_index][i_index]
                energy_consumption += energy_cost_i
                training_accuracy_i = training_accuracies[sel_index][i_index]
                training_accuracy += training_accuracy_i
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Tasks assignment: {0}".format(tasks_assignment))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))
        # print("Training accuracy: {0}".format(training_accuracy))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expect_number_selected_clients = 30
        self.assertEqual(expect_number_selected_clients, len(selected_clients))
        expected_tasks_assignment = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertSequenceEqual(expected_tasks_assignment, tasks_assignment)
        expected_makespan = 9.735729158265277
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 397.9426287817868
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.22868502471427568
        self.assertEqual(expected_training_accuracy, training_accuracy)

    def test_elastic_adapted_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 6  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        assignment_capacities_i = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        assignment_capacities = array(assignment_capacities)
        for i in range(I):
            assignment_capacities_i.append(list(range(0, assignment_capacities[i]+1)))
        # Monotonically increasing time costs.
        time_costs = array([[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)
        # Training accuracies set to zero (OLAR paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # Solution to ELASTIC's Adapted Algorithm 1: Client Selection.
        x, tasks_assignment, selected_clients \
            = elastic_adapted(I, assignment_capacities, time_costs, energy_costs, τ, α)
        makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for sel_index, num_tasks_scheduled in enumerate(list(tasks_assignment)):
            if num_tasks_scheduled > 0:
                i_index = assignment_capacities_i[sel_index].index(num_tasks_scheduled)
                time_cost_i = time_costs[sel_index][i_index]
                if time_cost_i > makespan:
                    makespan = time_cost_i
                energy_cost_i = energy_costs[sel_index][i_index]
                energy_consumption += energy_cost_i
                training_accuracy_i = training_accuracies[sel_index][i_index]
                training_accuracy += training_accuracy_i
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Tasks assignment: {0}".format(tasks_assignment))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))
        # print("Training accuracy: {0}".format(training_accuracy))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients))
        expected_tasks_assignment = [2, 2, 2]
        self.assertSequenceEqual(expected_tasks_assignment, tasks_assignment)
        expected_makespan = 10.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 0.0
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)

    def test_elastic_adapted_on_mc2mkp_paper_example_1(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 5  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        assignment_capacities_i = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        assignment_capacities = array(assignment_capacities)
        for i in range(I):
            assignment_capacities_i.append(list(range(0, assignment_capacities[i]+1)))
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # Solution to ELASTIC's Adapted Algorithm 1: Client Selection.
        x, tasks_assignment, selected_clients \
            = elastic_adapted(I, assignment_capacities, time_costs, energy_costs, τ, α)
        makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for sel_index, num_tasks_scheduled in enumerate(list(tasks_assignment)):
            if num_tasks_scheduled > 0:
                i_index = assignment_capacities_i[sel_index].index(num_tasks_scheduled)
                time_cost_i = time_costs[sel_index][i_index]
                if time_cost_i > makespan:
                    makespan = time_cost_i
                energy_cost_i = energy_costs[sel_index][i_index]
                energy_consumption += energy_cost_i
                training_accuracy_i = training_accuracies[sel_index][i_index]
                training_accuracy += training_accuracy_i
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Tasks assignment: {0}".format(tasks_assignment))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))
        # print("Training accuracy: {0}".format(training_accuracy))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients))
        expected_tasks_assignment = [3, 1, 1]
        self.assertSequenceEqual(expected_tasks_assignment, tasks_assignment)
        expected_makespan = 0.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 10
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)

    def test_elastic_adapted_on_mc2mkp_paper_example_2(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 8  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        assignment_capacities_i = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        assignment_capacities = array(assignment_capacities)
        for i in range(I):
            assignment_capacities_i.append(list(range(0, assignment_capacities[i]+1)))
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[0, 2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # Solution to ELASTIC's Adapted Algorithm 1: Client Selection.
        x, tasks_assignment, selected_clients \
            = elastic_adapted(I, assignment_capacities, time_costs, energy_costs, τ, α)
        makespan = 0
        energy_consumption = 0
        training_accuracy = 0
        for sel_index, num_tasks_scheduled in enumerate(list(tasks_assignment)):
            if num_tasks_scheduled > 0:
                i_index = assignment_capacities_i[sel_index].index(num_tasks_scheduled)
                time_cost_i = time_costs[sel_index][i_index]
                if time_cost_i > makespan:
                    makespan = time_cost_i
                energy_cost_i = energy_costs[sel_index][i_index]
                energy_consumption += energy_cost_i
                training_accuracy_i = training_accuracies[sel_index][i_index]
                training_accuracy += training_accuracy_i
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Tasks assignment: {0}".format(tasks_assignment))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))
        # print("Training accuracy: {0}".format(training_accuracy))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients))
        expected_tasks_assignment = [4, 2, 2]
        self.assertSequenceEqual(expected_tasks_assignment, tasks_assignment)
        expected_makespan = 0.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 14.5
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.0
        self.assertEqual(expected_training_accuracy, training_accuracy)
