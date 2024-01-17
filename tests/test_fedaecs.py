from math import pow
from numpy.random import uniform, seed
from numpy import array, int_, power
from unittest import TestCase
from schedulers.fedaecs import fedaecs


class TestFedAECS(TestCase):

    def test_fedaecs(self) -> None:
        # Execute the FedAECS algorithm.
        # FedAECS: Federated learning for Accuracy-Energy based Client Selection.
        # Goal: Balance the tradeoff between the energy consumption and the learning accuracy.

        # Seed.
        seed(7)

        # Simulation Parameters (Section VI, Subsection A).

        I = 1  # Number of rounds (I = 1000).

        K = 10  # Number of clients (K = {20, 40, 60, 80, 100}).

        B = 1.0 * pow(10, 6)  # Total bandwidth (B = {1, 3, 5, 7, 9} MHz, 1 MHz --> 1,000,000 Hz).

        # All client's data size info, which follow uniform distribution.
        # Each client k has a local dataset with Dik samples in epoch i.
        D = uniform(2, 10.001, (I, K)) * pow(10, 6)  # Dᵢₖ ∼ U(2, 10) MB, 1 MB --> 1,000,000 Bytes.
        D = array(list(map(int_, D)))

        # All client's bandwidth info, which follow uniform distribution.
        b = uniform(50, 150.001, (I, K)) * pow(10, 3)  # bᵢₖ ∼ U(50, 150) KHz, 1 KHz --> 1,000 Hz.

        # All client's transmission power info, which follow uniform distribution.
        P = uniform(4, 10.001, (I, K))  # Pᵢₖ ∼ U(4, 10) dBm.

        # The CPU frequency of client k.
        f = uniform(2, 4.001, K) * pow(10, 9)  # fₖ ∼ U(2, 4) GHz, 1 GHz --> 1.0e+09 Hz.

        # Number of CPU cycles which is used by client k to train the local model in one iteration.
        c = uniform(1, 3.001, K)  # cₖ ∼ U(1, 3) cycles/bit.

        S = 100 * pow(10, 3)  # Transmit data size (S = 100 kbits, 1kb --> 1000 bits).

        Uik = 10  # Number of local iterations (Uᵢₖ = 10).

        Vik = 4  # Number of global iterations (Vᵢₖ = 4).

        N0 = 1.0 * pow(10, -9)  # Noise power spectral density / channel noise (N0 = -90 dBm/Hz --> 1 x 10^−9.0 mW/Hz).
        # Decibel is a logarithmic unit. −174 dBm/Hz means 10^−17.4 mW/Hz.
        # https://dsp.stackexchange.com/questions/80854/conversion-of-dbm-hz-into-watt
        # https://3roam.com/dbm-hz-to-watt-and-watt-hz/

        γk = 1.0 * pow(10, -28)  # Effective switched capacitance in local computation (γₖ = 10⁻²⁸).

        # System parameter (μ = 1.7 x 10⁻⁸).
        # It must be adjusted according to the values of the D array.
        # Notice that, by definition, εik = log(1 + (μ * D[i][k])).
        # Therefore, 0 ≤ μ * D[i][k] ≤ 9, such that 0 ≤ εik ≤ 1, i.e., log(1) = 0 and log(10) = 1.
        μ = 1.7 * pow(10, -8)

        G = 40 * power((0.5 / uniform(0.6, 1.001, K)), 4)  # The channel gain between client k and the server.

        ε0 = 0.15  # The lower bound of accuracy (0.15).

        T_max = 1  # Deadline of a global iteration (T_max = 1 seconds).

        # Solution to FedAECS Algorithm.
        beta_star, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy \
            = fedaecs(I, K, Uik, Vik, ε0, T_max, B, S, N0, γk, μ, D, b, f, P, c, G)
        # for i in range(I):
        #     print("-------\nRound {0}:".format(i))
        #     print("βᵢ*: {0}".format(beta_star[i]))
        #     print("φᵢ(βᵢ*): {0}".format(f_obj_beta_star[i]))
        #     print("{0} (out of {1}) clients selected: {2}".format(len(selected_clients[i]), K, selected_clients[i]))
        #     print("Makespan (s): {0}".format(makespan[i]))
        #     print("Energy consumption (J): {0}".format(energy_consumption[i]))
        #     print("Training accuracy: {0}".format(training_accuracy[i]))

        # Asserts for the FedAECS algorithm results.
        expected_beta_star_0 = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.assertSequenceEqual(expected_beta_star_0, list(beta_star[0]))
        expected_f_obj_beta_star_0 = 19.391706095992046
        self.assertEqual(expected_f_obj_beta_star_0, f_obj_beta_star[0])
        expected_selected_clients_0 = [0, 1, 4]
        self.assertSequenceEqual(expected_selected_clients_0, selected_clients[0])
        expect_number_selected_clients = 3
        self.assertEqual(expect_number_selected_clients, len(selected_clients[0]))
        expected_makespan_0 = 0.5332146038795531
        self.assertEqual(expected_makespan_0, makespan[0])
        expected_energy_consumption_0 = 6.548116211244814
        self.assertEqual(expected_energy_consumption_0, energy_consumption[0])
        expected_training_accuracy_0 = 0.2844111743251012
        self.assertEqual(expected_training_accuracy_0, training_accuracy[0])
