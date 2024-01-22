from itertools import combinations
from math import inf, log, log2, pow
from numpy import any, argmin, array, ndarray, zeros


def calculate_E_comp(Uik: float,
                     γk: float,
                     ck: float,
                     Dik: float,
                     fk: float) -> float:
    return Uik * γk * ck * Dik * pow(fk, 2)


def calculate_E_up(Pik: float,
                   S: float,
                   bik: float,
                   Gk: float,
                   N0: float) -> float:
    return (Pik * S) / (bik * log2(1 + ((Pik * Gk) / (N0 * bik))))


def calculate_E(Vik: int,
                E_comp_ik: float,
                E_up_ik: float) -> float:
    return Vik * (E_comp_ik + E_up_ik)


def calculate_t_train(Uik: int,
                      ck: float,
                      Dik: float,
                      fk: float) -> float:
    return Uik * ((ck * Dik) / fk)


def calculate_R(bik: float,
                Pik: float,
                Gk: float,
                N0: float) -> float:
    return bik * log2(1 + ((Pik * Gk) / (N0 * bik)))


def calculate_t_up(S: float,
                   Rik: float) -> float:
    return S / Rik


def calculate_T(Vik: int,
                t_train_ik: float,
                t_up_ik: float) -> float:
    return Vik * (t_train_ik + t_up_ik)


def calculate_ε(μ: float,
                Dik: float) -> float:
    return log(1 + (μ * Dik))


def calculate_Γ(sum_εi: float) -> float:
    return log(1 + sum_εi)


def fedaecs(I: int,
            K: int,
            Uik: int,
            Vik: int,
            ε0: float,
            T_max: float,
            B: float,
            S: float,
            N0: float,
            γk: float,
            μ: float,
            D: ndarray,
            b: ndarray,
            f: ndarray,
            P: ndarray,
            c: ndarray,
            G: ndarray) -> tuple:
    beta_star = []
    f_obj_beta_star = []
    selected_clients = []
    makespan = []
    energy_consumption = []
    training_accuracy = []
    # For each communication round i.
    for i in range(I):
        εi = []
        Ei = []
        Ti = []
        for k in range(K):
            # Calculate the training accuracy of client k.
            ε_ik = calculate_ε(μ, D[i][k])
            εi.append(ε_ik)
            # Calculate the energy consumption of client k to compute its local model.
            E_comp_ik = calculate_E_comp(Uik, γk, c[k], D[i][k], f[k])
            # Calculate the energy consumption of client k to upload its local model.
            E_up_ik = calculate_E_up(P[i][k], S, b[i][k], G[k], N0)
            # Calculate the total energy consumption of client k (Eᵢₖ).
            E_ik = calculate_E(Vik, E_comp_ik, E_up_ik)
            Ei.append(E_ik)
            # Calculate the required time by client k to compute its local model.
            t_train_ik = calculate_t_train(Uik, c[k], D[i][k], f[k])
            # Calculate the required time by client k to upload its local model.
            R_ik = calculate_R(b[i][k], P[i][k], G[k], N0)
            t_up_ik = calculate_t_up(S, R_ik)
            # Calculate the total time of client k (Tᵢₖ).
            T_ik = calculate_T(Vik, t_train_ik, t_up_ik)
            Ti.append(T_ik)
        # Auxiliary variables.
        n = []
        init_qualified_client_energy = []
        init_qualified_client_accuracy = []
        init_qualified_client_time = []
        init_qualified_client_bandwidth = []
        init_qualified_client_index = []
        init_unqualified_client_index = []
        # Optimization variables.
        beta_line_i = zeros(shape=K, dtype=int)
        beta_star_i = zeros(shape=K, dtype=int)
        f_obj_beta_star_i = inf
        # Check the bandwidth and time constraints (preliminary screening).
        for k in range(K):
            if Ti[k] <= T_max and b[i][k] <= B:
                n.append(Ei[k] / εi[k])
                init_qualified_client_energy.append(Ei[k])
                init_qualified_client_accuracy.append(εi[k])
                init_qualified_client_time.append(Ti[k])
                init_qualified_client_bandwidth.append(b[i][k])
                init_qualified_client_index.append(k)
            else:
                init_unqualified_client_index.append(k)
        # Output the unqualified clients.
        if init_unqualified_client_index:
            for unqualified_client_index in init_unqualified_client_index:
                beta_line_i[unqualified_client_index] = 1
        # Sort n in ascending order (according to the ratio of the energy consumption and the FL accuracy).
        sorted_client_n = []
        sorted_client_energy = []
        sorted_client_accuracy = []
        sorted_client_bandwidth = []
        sorted_client_index = []
        if init_qualified_client_index:
            (sorted_client_n,
             sorted_client_energy,
             sorted_client_accuracy,
             sorted_client_bandwidth,
             sorted_client_index) \
                = map(list, zip(*sorted(zip(n,
                                            init_qualified_client_energy,
                                            init_qualified_client_accuracy,
                                            init_qualified_client_bandwidth,
                                            init_qualified_client_index),
                                        reverse=False)))
        # Initializing m.
        m = 0
        while m <= len(sorted_client_n) - 1:
            if sorted_client_accuracy[m] >= ε0:
                client_index = sorted_client_index[m]
                beta_star_i[client_index] = 1
                f_obj_beta_star_i = sorted_client_n[m]
                break
            else:
                if m <= len(sorted_client_n) - 1:
                    m = m + 1
                    while 1 <= m <= len(sorted_client_n) - 1:
                        if sorted_client_accuracy[m] >= ε0:
                            client_index = sorted_client_index[m]
                            beta_star_i[client_index] = 1
                            the_first_qualified_client_index = beta_star_i
                            # Check the combination selection of the previous clients.
                            selection_possibilities = []
                            for t in range(0, m):
                                s = array(list(combinations(range(0, m), t)))
                                if not any(s):
                                    continue
                                for si in s:
                                    selection_possibilities.append(list(si))
                            selection_possibilities.append(list(range(0, m)))
                            qualified_selection = []
                            obj = []
                            # Calculate the model accuracy and total bandwidth for each selection possibility.
                            for selection_possibility in selection_possibilities:
                                sum_accuracy_select_idx = 0
                                sum_bandwidth_select_idx = 0
                                for client_idx in selection_possibility:
                                    sum_accuracy_select_idx += sorted_client_accuracy[client_idx]
                                    sum_bandwidth_select_idx += sorted_client_bandwidth[client_idx]
                                model_accuracy_select_idx = calculate_Γ(sum_accuracy_select_idx)
                                # Check the constraints are whether satisfied.
                                if (model_accuracy_select_idx >= ε0 and
                                        sum_bandwidth_select_idx <= B):
                                    # Calculate the total energy consumption of the qualified selection.
                                    total_energy_qualified_select_idx = 0
                                    for client_idx in selection_possibility:
                                        total_energy_qualified_select_idx += sorted_client_energy[client_idx]
                                    # Calculate the objective function.
                                    f_obj = (total_energy_qualified_select_idx /
                                             model_accuracy_select_idx)
                                    obj = list(obj)
                                    # Store the objective function value.
                                    obj.append(f_obj)
                                    # Store the qualified selection.
                                    qualified_selection.append(selection_possibility)
                            obj = array(obj)
                            # Check whether there is a client selection for combinatorial optimization
                            # satisfying constraints.
                            if qualified_selection:
                                # y is the location (index) of objective function minimum value.
                                y = argmin(obj)
                                # Further compare the optimal values for the objective function.
                                if obj[y] <= sorted_client_n[m]:
                                    f_obj_beta_star_i = obj[y]
                                    for qs_idx in qualified_selection[y]:
                                        beta_star_i[qs_idx] = 1
                                else:
                                    f_obj_beta_star_i = sorted_client_n[m]
                                    beta_star_i = the_first_qualified_client_index
                            else:
                                beta_star_i = the_first_qualified_client_index
                            break
                        else:
                            m = m + 1
        # Organizing the solution for the round i.
        selected_clients_i = []
        makespan_i = 0
        sum_energy_consumption_i = 0
        sum_accuracy_i = 0
        for client_index, _ in enumerate(beta_star_i):
            if beta_star_i[client_index] == 1:
                selected_clients_i.append(client_index)
                makespan_ik = Ti[client_index]
                if makespan_ik > makespan_i:
                    makespan_i = makespan_ik
                energy_consumption_ik = Ei[client_index]
                sum_energy_consumption_i += energy_consumption_ik
                accuracy_ik = εi[client_index]
                sum_accuracy_i += accuracy_ik
        beta_star.append(beta_star_i)
        f_obj_beta_star.append(f_obj_beta_star_i)
        selected_clients.append(selected_clients_i)
        makespan.append(makespan_i)
        energy_consumption.append(sum_energy_consumption_i)
        accuracy_i = calculate_Γ(sum_accuracy_i)
        training_accuracy.append(accuracy_i)
    return beta_star, f_obj_beta_star, selected_clients, makespan, energy_consumption, training_accuracy
