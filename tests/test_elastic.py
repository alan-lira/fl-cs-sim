from math import pow, sqrt
from numpy.random import randint, uniform, seed
from numpy import array
from unittest import TestCase
from schedulers.elastic import elastic_client_selection_algorithm


class TestELASTIC(TestCase):

    def test_elastic(self) -> None:

        # Execute the ELASTIC algorithm.
        # ELASTIC: Energy and Latency-Aware Resource Management and Client Selection for Federated Learning.
        # Goal: Optimize the tradeoff between maximizing the number of selected clients and
        # minimizing the total energy consumption.

        # Seed.
        seed(7)

        # Simulation Parameters (Section V).

        I = 30  # 30 clients uniformly distributed in a 2 km x 2 km area covered by a BS.
        m = 2
        n = 2

        xBS = m / 2  # BS positioned at the center of the covered area.
        yBS = n / 2

        d = []
        x = uniform(0, m + 0.001, I)  # xᵢ ∼ U(0, m)
        y = uniform(0, n + 0.001, I)  # yᵢ ∼ U(0, n)
        for i in range(I):
            di = sqrt(pow(xBS - x[i], 2) + pow(yBS - y[i], 2))  # Distance in kilometer between the BS and client i.
            d.append(di)
        d = array(d)

        g = []
        for i in range(I):
            gi = 128.1 + 37.6 * d[i]  # Path loss between the BS and a client.
            g.append(gi)
        g = array(g)

        D = []
        for i in range(I):
            Di = 200  # Each client would train its local model over |Dᵢ| = 200 samples in each global iteration.
            D.append(Di)
        D = array(D)

        # Average number of CPU cycles required for training one data sample (one per client).
        C = randint(3, 6, I) * pow(10, 5)  # Cᵢ ∼ U(3, 5) × 10⁵ CPU cycles/sample.

        # The minimum CPU frequency of a client fᵢ_min.
        f_min = uniform(0.2, 1.001, I) * pow(10, 9)  # fᵢ_min ∼ U(0.2, 1) × 10 GHz (1 GHz --> 1,000,000,000 Hz).

        f_max = []  # The maximum CPU frequency of a client fᵢ_max.
        for i in range(I):
            fi_max = f_min[i] + (2 * pow(10, 9))  # fᵢ_max = fᵢ_min + 2 GHz ∀i ∈ I.
            f_max.append(fi_max)
        f_max = array(f_max)

        p_max = 1.0  # Maximum transmission power (p_max = 1 Watt).

        N0 = 1.0 * pow(10, -9)  # Noise and inter-cell interference (N0 = -90 dBm/Hz --> 1 x 10^−9.0 mW/Hz).
        # Decibel is a logarithmic unit. −174 dBm/Hz means 10^−17.4 mW/Hz.
        # https://dsp.stackexchange.com/questions/80854/conversion-of-dbm-hz-into-watt
        # https://3roam.com/dbm-hz-to-watt-and-watt-hz/

        B = 1.0 * pow(10, 6)  # Bandwidth (B = 1 MHz --> 1,000,000 Hz).

        s = 500 * pow(10, 3)  # Size of the local model (s = 500 kbits, 1kb --> 1000 bits).

        θ = 1  # Constant determined by the desired global model (θ = 1).

        ε = 0.05  # Desired accuracy (ε = 0.05).

        τ = 10  # Deadline of a global iteration (τ = 10 seconds).

        γ = 1.0 * pow(10, -28)  # Switch capacitance coefficient in Eq. (7) (γ = 10⁻²⁸).

        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.

        training_accuracies = []  # Training accuracies.
        for i in range(I):
            training_accuracy_i = uniform(0, 0.015)  # εᵢ ∼ U(0, 0.15).
            training_accuracies.append(training_accuracy_i)

        # Solution to ELASTIC's Algorithm 1: Client Selection.
        x, selected_clients, makespan, energy_consumption \
            = elastic_client_selection_algorithm(I, g, D, C, f_max, p_max, N0, B, s, θ, ε, τ, γ, α)
        training_accuracy = 0
        for index, value in enumerate(list(x)):
            if value > 0:
                training_accuracy_i = training_accuracies[index]
                training_accuracy += training_accuracy_i
        # print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients), I, selected_clients))
        # print("Makespan (s): {0}".format(makespan))
        # print("Energy consumption (J): {0}".format(energy_consumption))
        # print("Training accuracy: {0}".format(training_accuracy))

        # Asserts for the ELASTIC algorithm results.
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
        expected_makespan = 0.4357632632271027
        self.assertEqual(expected_makespan, makespan)
        expected_energy_consumption = 7.074139733231213
        self.assertEqual(expected_energy_consumption, energy_consumption)
        expected_training_accuracy = 0.24158314367430217
        self.assertEqual(expected_training_accuracy, training_accuracy)
