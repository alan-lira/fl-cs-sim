from numpy import array, inf
from numpy.random import randint, seed, uniform
from unittest import TestCase
from schedulers.fedaecs_adapted import fedaecs_adapted


class TestFedAECSAdapted(TestCase):

    def test_fedaecs_adapted(self) -> None:
        # Execute the adapted version of FedAECS algorithm.
        # FedAECS: Federated learning for Accuracy-Energy based Client Selection.
        # Goal: Balance the tradeoff between the energy consumption and the learning accuracy.

        # Seed.
        seed(7)

        # Simulation Parameters.

        I = 1  # 1 round.

        K = 10  # 10 clients.

        num_tasks = 10  # 10 tasks.

        assignment_capacities = []  # Task assignment capacities per resource per round.
        time_costs = []  # Time costs to process tasks per resource per round.
        energy_costs = []  # Energy costs to process tasks per resource per round.
        training_accuracies = []  # Training accuracies per resource per round.
        b = []  # Bandwidth information per resource per round.

        for i in range(I):

            A_i = []
            for k in range(K):
                max_A_ik = randint(1, num_tasks+1)
                A_ika = []
                for a in range(1, max_A_ik+1):
                    A_ika.append(a)
                A_i.append(A_ika)
            assignment_capacities.append(A_i)

            T_i = []
            for k in range(K):
                T_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    # tᵢₖₐ ∼ U(0.15, 0.25) × Aᵢₖₐ seconds.
                    T_ika = assignment_capacities[i][k][a] * uniform(0.15, 0.251)
                    T_ik.append(T_ika)
                T_i.append(T_ik)
            time_costs.append(T_i)

            E_i = []
            for k in range(K):
                E_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    # Eᵢₖₐ ∼ U(0, 5) × tᵢₖₐ × 10⁻² Joules.
                    E_ika = time_costs[i][k][a] * uniform(0, 5.001) * pow(10, -2)
                    E_ik.append(E_ika)
                E_i.append(E_ik)
            energy_costs.append(E_i)

            ε_i = []
            for k in range(K):
                ε_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    # εᵢₖₐ ∼ U(0, 0.15).
                    ε_ika = uniform(0, 0.15)
                    ε_ik.append(ε_ika)
                ε_i.append(ε_ik)
            training_accuracies.append(ε_i)

            b_i = []
            for k in range(K):
                b_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    # bᵢₖₐ = 0.
                    b_ika = 0
                    b_ik.append(b_ika)
                b_i.append(b_ik)
            b.append(b_i)

        ε0 = 0.1  # The lower bound of accuracy (0.1).

        T_max = 1  # Deadline of a global iteration (T_max = 1 seconds).

        B = inf  # Total bandwidth (B = ∞ Hz).

        # Solution to FedAECS Adapted Algorithm.
        beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy \
            = fedaecs_adapted(I, K, num_tasks, assignment_capacities, time_costs, energy_costs, training_accuracies,
                              b, ε0, T_max, B)
        # for i in range(I):
        #     print("-------\nRound {0}:".format(i))
        #     print("βᵢ*: {0}".format(beta_star[i]))
        #     print("βᵢ*_tasks: {0}".format(beta_star_tasks[i]))
        #     print("φᵢ(βᵢ*): {0}".format(f_obj_beta_star[i]))
        #     print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients[i]), K, selected_clients[i]))
        #     print("Makespan (s): {0}".format(makespan[i]))
        #     print("Energy consumption (J): {0}".format(energy_consumption[i]))
        #     print("Training accuracy: {0}".format(training_accuracy[i]))

        # Asserts for the FedAECS algorithm results.
        expected_beta_star_0 = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
        self.assertSequenceEqual(expected_beta_star_0, list(beta_star[0]))
        expected_beta_star_tasks_0 = [0, 2, 3, 0, 0, 4, 0, 0, 0, 1]
        self.assertSequenceEqual(expected_beta_star_tasks_0, list(beta_star_tasks[0]))
        expected_f_obj_beta_star_0 = 0.046802240623087195
        self.assertEqual(expected_f_obj_beta_star_0, f_obj_beta_star[0])
        expected_selected_clients_0 = [1, 2, 5, 9]
        self.assertSequenceEqual(expected_selected_clients_0, selected_clients[0])
        expect_number_selected_clients = 4
        self.assertEqual(expect_number_selected_clients, len(selected_clients[0]))
        expected_makespan_0 = 0.6896013440077446
        self.assertEqual(expected_makespan_0, makespan[0])
        expected_energy_consumption_0 = 0.0461461021308035
        self.assertEqual(expected_energy_consumption_0, energy_consumption[0])
        expected_training_accuracy_0 = 0.31730065672703156
        self.assertEqual(expected_training_accuracy_0, training_accuracy[0])

    def test_fedaecs_adapted_on_olar_paper_example(self) -> None:
        # Number of rounds, resources, and tasks.
        num_rounds = 1
        num_resources = 3
        num_tasks = 6
        # Task assignment capacities per resource.
        assignment_capacities = array([[[0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [0, 1, 2, 3, 4, 5, 6]]], dtype=object)
        # Monotonically increasing time costs.
        time_costs = array([[[0.5, 2, 4, 7, 9, 11, 14], [0, 1, 3, 5, 7, 9, 11], [1, 6, 10, 15, 22, 23, 27]]],
                           dtype=object)
        # Energy costs set to zero (OLAR paper's example doesn't consider them).
        energy_costs = array([[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], dtype=object)
        # Training accuracies set to zero (OLAR paper's example doesn't consider them).
        training_accuracies = array([[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], dtype=object)

        # Bandwidth information per resource per round.
        b = []
        for i in range(num_rounds):
            b_i = []
            for k in range(num_resources):
                b_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    b_ika = 0
                    b_ik.append(b_ika)
                b_i.append(b_ik)
            b.append(b_i)

        ε0 = 0.0  # The lower bound of accuracy (0.0).

        T_max = 50  # Deadline of a global iteration (T_max = 50 seconds).

        B = inf  # Total bandwidth (B = ∞ Hz).

        # Solution to FedAECS Adapted Algorithm.
        beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy \
            = fedaecs_adapted(num_rounds, num_resources, num_tasks, assignment_capacities, time_costs, energy_costs,
                              training_accuracies, b, ε0, T_max, B)
        # for i in range(I):
        #     print("-------\nRound {0}:".format(i))
        #     print("βᵢ*: {0}".format(beta_star[i]))
        #     print("βᵢ*_tasks: {0}".format(beta_star_tasks[i]))
        #     print("φᵢ(βᵢ*): {0}".format(f_obj_beta_star[i]))
        #     print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients[i]), K, selected_clients[i]))
        #     print("Makespan (s): {0}".format(makespan[i]))
        #     print("Energy consumption (J): {0}".format(energy_consumption[i]))
        #     print("Training accuracy: {0}".format(training_accuracy[i]))

        # Asserts for the FedAECS algorithm results.
        expected_beta_star_0 = [1, 1, 1]
        self.assertSequenceEqual(expected_beta_star_0, list(beta_star[0]))
        expected_beta_star_tasks_0 = [3, 2, 1]
        self.assertSequenceEqual(expected_beta_star_tasks_0, list(beta_star_tasks[0]))
        expected_f_obj_beta_star_0 = inf
        self.assertEqual(expected_f_obj_beta_star_0, f_obj_beta_star[0])
        expected_selected_clients_0 = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients_0, selected_clients[0])
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients[0]))
        expected_makespan_0 = 7.0
        self.assertEqual(expected_makespan_0, makespan[0])
        expected_energy_consumption_0 = 0.0
        self.assertEqual(expected_energy_consumption_0, energy_consumption[0])
        expected_training_accuracy_0 = 0.0
        self.assertEqual(expected_training_accuracy_0, training_accuracy[0])

    def test_fedaecs_adapted_on_mc2mkp_paper_example_1(self) -> None:
        # Number of rounds, resources, and tasks.
        num_rounds = 1
        num_resources = 3
        num_tasks = 5
        # Task assignment capacities per resource.
        assignment_capacities = array([[[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]], dtype=object)
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[[2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], dtype=object)

        # Bandwidth information per resource per round.
        b = []
        for i in range(num_rounds):
            b_i = []
            for k in range(num_resources):
                b_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    b_ika = 0
                    b_ik.append(b_ika)
                b_i.append(b_ik)
            b.append(b_i)

        ε0 = 0.0  # The lower bound of accuracy (0.0).

        T_max = 50  # Deadline of a global iteration (T_max = 50 seconds).

        B = inf  # Total bandwidth (B = ∞ Hz).

        # Solution to FedAECS Adapted Algorithm.
        beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy \
            = fedaecs_adapted(num_rounds, num_resources, num_tasks, assignment_capacities, time_costs, energy_costs,
                              training_accuracies, b, ε0, T_max, B)
        # for i in range(I):
        #     print("-------\nRound {0}:".format(i))
        #     print("βᵢ*: {0}".format(beta_star[i]))
        #     print("βᵢ*_tasks: {0}".format(beta_star_tasks[i]))
        #     print("φᵢ(βᵢ*): {0}".format(f_obj_beta_star[i]))
        #     print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients[i]), K, selected_clients[i]))
        #     print("Makespan (s): {0}".format(makespan[i]))
        #     print("Energy consumption (J): {0}".format(energy_consumption[i]))
        #     print("Training accuracy: {0}".format(training_accuracy[i]))

        # Asserts for the FedAECS algorithm results.
        expected_beta_star_0 = [1, 1, 1]
        self.assertSequenceEqual(expected_beta_star_0, list(beta_star[0]))
        expected_beta_star_tasks_0 = [1, 3, 1]
        self.assertSequenceEqual(expected_beta_star_tasks_0, list(beta_star_tasks[0]))
        expected_f_obj_beta_star_0 = inf
        self.assertEqual(expected_f_obj_beta_star_0, f_obj_beta_star[0])
        expected_selected_clients_0 = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients_0, selected_clients[0])
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients[0]))
        expected_makespan_0 = 0.0
        self.assertEqual(expected_makespan_0, makespan[0])
        expected_energy_consumption_0 = 9.0
        self.assertEqual(expected_energy_consumption_0, energy_consumption[0])
        expected_training_accuracy_0 = 0.0
        self.assertEqual(expected_training_accuracy_0, training_accuracy[0])

    def test_fedaecs_adapted_on_mc2mkp_paper_example_2(self) -> None:
        # Number of rounds, resources, and tasks.
        num_rounds = 1
        num_resources = 3
        num_tasks = 8
        # Task assignment capacities per resource.
        assignment_capacities = array([[[1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]], dtype=object)
        # Time costs set to zero (MC²MKP paper's example doesn't consider them).
        time_costs = array([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], dtype=object)
        # Monotonically increasing energy costs.
        energy_costs = array([[[2, 3.5, 5.5, 8, 10, 12], [0, 1.5, 2.5, 4, 7, 9, 11], [0, 3, 4, 5, 6, 7]]],
                             dtype=object)
        # Training accuracies set to zero (MC²MKP paper's example doesn't consider them).
        training_accuracies = array([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], dtype=object)

        # Bandwidth information per resource per round.
        b = []
        for i in range(num_rounds):
            b_i = []
            for k in range(num_resources):
                b_ik = []
                for a in range(len(assignment_capacities[i][k])):
                    b_ika = 0
                    b_ik.append(b_ika)
                b_i.append(b_ik)
            b.append(b_i)

        ε0 = 0.0  # The lower bound of accuracy (0.0).

        T_max = 50  # Deadline of a global iteration (T_max = 50 seconds).

        B = inf  # Total bandwidth (B = ∞ Hz).

        # Solution to FedAECS Adapted Algorithm.
        beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy \
            = fedaecs_adapted(num_rounds, num_resources, num_tasks, assignment_capacities, time_costs, energy_costs,
                              training_accuracies, b, ε0, T_max, B)
        # for i in range(I):
        #     print("-------\nRound {0}:".format(i))
        #     print("βᵢ*: {0}".format(beta_star[i]))
        #     print("βᵢ*_tasks: {0}".format(beta_star_tasks[i]))
        #     print("φᵢ(βᵢ*): {0}".format(f_obj_beta_star[i]))
        #     print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients[i]), K, selected_clients[i]))
        #     print("Makespan (s): {0}".format(makespan[i]))
        #     print("Energy consumption (J): {0}".format(energy_consumption[i]))
        #     print("Training accuracy: {0}".format(training_accuracy[i]))

        # Asserts for the FedAECS algorithm results.
        expected_beta_star_0 = [1, 1, 1]
        self.assertSequenceEqual(expected_beta_star_0, list(beta_star[0]))
        expected_beta_star_tasks_0 = [3, 3, 2]
        self.assertSequenceEqual(expected_beta_star_tasks_0, list(beta_star_tasks[0]))
        expected_f_obj_beta_star_0 = inf
        self.assertEqual(expected_f_obj_beta_star_0, f_obj_beta_star[0])
        expected_selected_clients_0 = [0, 1, 2]
        self.assertSequenceEqual(expected_selected_clients_0, selected_clients[0])
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients[0]))
        expected_makespan_0 = 0.0
        self.assertEqual(expected_makespan_0, makespan[0])
        expected_energy_consumption_0 = 13.5
        self.assertEqual(expected_energy_consumption_0, energy_consumption[0])
        expected_training_accuracy_0 = 0.0
        self.assertEqual(expected_training_accuracy_0, training_accuracy[0])
