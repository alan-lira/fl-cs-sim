from numpy import array
from numpy.random import uniform, seed
from unittest import TestCase
from schedulers.elastic_adapted import elastic_adapted_client_selection_algorithm


class TestELASTICAdapted(TestCase):

    def test_elastic_adapted(self) -> None:

        # Execute the adapted version of ELASTIC algorithm.
        # ELASTIC: Energy and Latency-Aware Resource Management and Client Selection for Federated Learning.
        # Goal: Optimize the tradeoff between maximizing the number of selected clients and
        # minimizing the total energy consumption.

        # Seed.
        seed(7)

        # Simulation Parameters.

        I = 30  # 30 clients.

        T = 30  # 30 tasks.

        assignment_capacities = []  # Task assignment capacities per resource.
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover

        time_costs = []  # Time costs to process tasks per resource.
        for i in range(I):
            time_costs_i = uniform(0, 10.001, T)  # tᵢ ∼ U(0, 10) seconds.
            time_costs.append(time_costs_i)

        energy_costs = []  # Energy costs to process tasks per resource.
        for i in range(I):
            energy_costs_i = []
            for t in range(T):
                energy_cost_i = time_costs[i][t] * uniform(0, 5.001)  # Eᵢ ∼ U(0, 5) × tᵢ Joules.
                energy_costs_i.append(energy_cost_i)
            energy_costs.append(energy_costs_i)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # ELASTIC's Adapted Algorithm 1: Client Selection.
        x, selected_clients, makespan, energy_consumption \
            = elastic_adapted_client_selection_algorithm(I, assignment_capacities, time_costs, energy_costs, τ, α)
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [23, 17, 25, 27, 5, 19, 0, 13, 29, 20,
                                     1, 12, 21, 24, 4, 7, 9, 22, 26, 6,
                                     15, 3, 14, 16, 8, 28, 18, 10, 11, 2]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expected_makespan = 9.735729158265277
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 352.93469393438716
        self.assertEqual(expected_energy_consumption, energy_consumption)

    def test_elastic_adapted_on_olar_paper_example(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 6  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        # The cost arrays doesn't contain the costs of scheduling 0 tasks.
        # Monotonically increasing time costs.
        time_costs = array([[2, 4, 7, 9, 11, 14], [1, 3, 5, 7, 9, 11], [6, 10, 15, 22, 23, 27]],
                           dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # ELASTIC's Adapted Algorithm 1: Client Selection.
        x, selected_clients, makespan, energy_consumption \
            = elastic_adapted_client_selection_algorithm(I, assignment_capacities, time_costs, energy_costs, τ, α)
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expected_makespan = 10.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 0.0
        self.assertEqual(expected_energy_consumption, energy_consumption)

    def test_elastic_adapted_on_mc2mkp_paper_example_1(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 5  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        # The cost arrays doesn't contain the costs of scheduling 0 tasks.
        # Time costs set to zero (MC^2MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[2, 3.5, 5.5, 8, 10, 12], [1.5, 2.5, 4, 7, 9, 11], [3, 4, 5, 6, 7, 99]],
                             dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # ELASTIC's Adapted Algorithm 1: Client Selection.
        x, selected_clients, makespan, energy_consumption \
            = elastic_adapted_client_selection_algorithm(I, assignment_capacities, time_costs, energy_costs, τ, α)
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [1, 2, 0]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expected_makespan = 0.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 10.0
        self.assertEqual(expected_energy_consumption, energy_consumption)

    def test_elastic_adapted_on_mc2mkp_paper_example_2(self) -> None:
        # Number of resources and tasks.
        I = 3  # num_resources
        T = 8  # num_tasks
        # Task assignment capacities per resource.
        assignment_capacities = []
        # Divide the tasks as equally as possible.
        mean_tasks = T // I
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = T % I
        for i in range(I):
            Ai = mean_tasks
            assignment_capacities.append(Ai)
        assignment_capacities[0] += leftover
        # The cost arrays doesn't contain the costs of scheduling 0 tasks.
        # Time costs set to zero (MC^2MKP paper's example doesn't consider them).
        time_costs = array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[2, 3.5, 5.5, 8, 10, 12], [1.5, 2.5, 4, 7, 9, 11], [3, 4, 5, 6, 7, 99]],
                             dtype=object)

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        # ELASTIC's Adapted Algorithm 1: Client Selection.
        x, selected_clients, makespan, energy_consumption \
            = elastic_adapted_client_selection_algorithm(I, assignment_capacities, time_costs, energy_costs, τ, α)
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))

        # Asserts for the ELASTIC adapted algorithm results.
        expected_x = [1, 1, 1]
        self.assertSequenceEqual(expected_x, list(x))
        expected_selected_clients = [1, 2, 0]
        self.assertSequenceEqual(expected_selected_clients, selected_clients)
        expected_makespan = 0.0
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 14.5
        self.assertEqual(expected_energy_consumption, energy_consumption)
