from itertools import combinations
from math import inf, log
from numpy import any, argmin, array, ndarray, zeros


def calculate_Γ(sum_εi: float) -> float:
    return log(1 + sum_εi)


def fedaecs_adapted(I: int,
                    K: int,
                    num_tasks: int,
                    A: ndarray,
                    T: ndarray,
                    E: ndarray,
                    ε: ndarray,
                    b: ndarray,
                    ε0: float,
                    T_max: float,
                    B: float) -> tuple:
    # Some remarks about this adapted version of FedAECS algorithm:
    # 1. We considered that clients does not share a channel with limited bandwidth (B = ∞),
    #    so it doesn't matter their bandwidth information. In other words, ∀i ∈ I ∀k ∈ K, bik = 0.
    # 2. The algorithm receives the total number of tasks to be allocated among the selected clients (num_tasks),
    #    such that the algorithm won't stop until this constraint is also respected.
    # 3. The algorithm receives a previously generated array of task assignment capacities (A), ∀i ∈ I ∀k ∈ K,
    #    such that at the round i the client k can process between Aik_min (inclusive) and Aik_max (inclusive) tasks.
    #    Note that Aik specifies the number of tasks while Dik originally specified the size of tasks.
    # 4. The algorithm receives a previously generated array of time costs (T),
    #    such that Tika = t_train_ika + t_up_ika, ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    # 5. The algorithm receives a previously generated array of energy costs (E),
    #    such that Eika = E_comp_ika + E_up_ika, ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    # 6. The algorithm receives a previously generated array of training accuracies (ε), ∀i ∈ I ∀k ∈ K ∀a ∈ A.
    beta_star = []
    beta_star_tasks = []
    f_obj_beta_star = []
    selected_clients = []
    makespan = []
    energy_consumption = []
    training_accuracy = []
    # For each communication round i.
    for i in range(I):
        # Get the training accuracy of clients.
        εi = ε[i]
        # Get the total energy consumption (E_comp_ik + E_up_ik) of clients.
        Ei = E[i]
        # Get the total time (t_train_ik + t_up_ik) of clients.
        Ti = T[i]
        # Auxiliary variables.
        n = []
        init_qualified_client_capacity = []
        init_qualified_client_max_capacity = []
        init_qualified_client_energy = []
        init_qualified_client_accuracy = []
        init_qualified_client_time = []
        init_qualified_client_bandwidth = []
        init_qualified_client_index = []
        init_unqualified_client_index = []
        obj = []
        selection_possibilities = []
        selection_possibilities_model_accuracies = []
        selection_possibilities_total_bandwidths = []
        selection_possibilities_num_tasks = []
        qualified_selection = []
        # Optimization variables.
        beta_line_i = zeros(shape=K, dtype=int)
        beta_star_i = zeros(shape=K, dtype=int)
        beta_star_tasks_i = zeros(shape=K, dtype=int)
        f_obj_beta_star_i = inf
        # Check the bandwidth and time constraints (preliminary screening).
        for k in range(K):
            Ak = A[i][k]
            Ak_max = max(Ak)
            init_qualified_client_max_capacity.append(Ak_max)
            for a in Ak:
                if Ti[k][a-1] <= T_max and b[i][k][a-1] <= B:
                    if εi[k][a-1] > 0:
                        n.append(Ei[k][a-1] / εi[k][a-1])
                    else:
                        n.append(inf)
                    init_qualified_client_capacity.append(a)
                    init_qualified_client_energy.append(Ei[k][a-1])
                    init_qualified_client_accuracy.append(εi[k][a-1])
                    init_qualified_client_time.append(Ti[k][a-1])
                    init_qualified_client_bandwidth.append(b[i][k][a-1])
                    init_qualified_client_index.append(k)
                else:
                    init_unqualified_client_index.append(k)
        # Output the unqualified clients.
        if init_unqualified_client_index:
            for unqualified_client_index in init_unqualified_client_index:
                beta_line_i[unqualified_client_index] = 1
        # Sort n in ascending order (according to the ratio of the energy consumption and the FL accuracy).
        sorted_client_n = []
        sorted_client_capacity = []
        sorted_client_energy = []
        sorted_client_accuracy = []
        sorted_client_bandwidth = []
        sorted_client_index = []
        if init_qualified_client_index:
            (sorted_client_n,
             sorted_client_capacity,
             sorted_client_energy,
             sorted_client_accuracy,
             sorted_client_bandwidth,
             sorted_client_index) \
                = map(list, zip(*sorted(zip(n,
                                            init_qualified_client_capacity,
                                            init_qualified_client_energy,
                                            init_qualified_client_accuracy,
                                            init_qualified_client_bandwidth,
                                            init_qualified_client_index),
                                        reverse=False)))
        # Initializing m.
        m = 0
        num_tasks_assigned = 0
        while m <= len(sorted_client_n) - 1:
            if sorted_client_accuracy[m] >= ε0 and (sorted_client_capacity[m] + num_tasks_assigned) >= num_tasks:
                client_index = sorted_client_index[m]
                beta_star_i[client_index] = 1
                tasks_to_assign = num_tasks - num_tasks_assigned
                beta_star_tasks_i[client_index] += tasks_to_assign
                num_tasks_assigned += tasks_to_assign
                f_obj_beta_star_i = sorted_client_n[m]
                break
            else:
                if m <= len(sorted_client_n) - 1:
                    m = m + 1
                    while 1 <= m <= len(sorted_client_n) - 1:
                        if sorted_client_accuracy[m] >= ε0:
                            client_index = sorted_client_index[m]
                            if beta_star_tasks_i[client_index] + sorted_client_capacity[m] \
                                    <= init_qualified_client_max_capacity[client_index]:
                                tasks_to_assign = sorted_client_capacity[m]
                                if num_tasks_assigned + tasks_to_assign > num_tasks:
                                    tasks_to_assign = num_tasks - num_tasks_assigned
                                beta_star_i[client_index] = 1
                                beta_star_tasks_i[client_index] += tasks_to_assign
                                num_tasks_assigned += tasks_to_assign
                            else:
                                tasks_to_assign \
                                    = init_qualified_client_max_capacity[client_index] - beta_star_tasks_i[client_index]
                                if num_tasks_assigned + tasks_to_assign > num_tasks:
                                    tasks_to_assign = num_tasks - num_tasks_assigned
                                beta_star_i[client_index] = 1
                                beta_star_tasks_i[client_index] += tasks_to_assign
                                num_tasks_assigned += tasks_to_assign
                            the_first_qualified_client_index = beta_star_i
                            the_first_qualified_client_index_tasks = beta_star_tasks_i
                            # Check the combination selection of the previous clients.
                            for t in range(0, m):
                                s = array(list(combinations(range(0, m), t)))
                                if not any(s):
                                    continue
                                for si in s:
                                    selection_possibilities.append(list(si))
                            selection_possibilities.append(list(range(0, m)))
                            # Calculate the model accuracy and total bandwidth for each selection possibility.
                            for selection_possibility in selection_possibilities:
                                sum_accuracy_select_idx = 0
                                sum_bandwidth_select_idx = 0
                                sum_tasks_select_idx = 0
                                for client_idx in selection_possibility:
                                    sum_accuracy_select_idx += sorted_client_accuracy[client_idx]
                                    sum_bandwidth_select_idx += sorted_client_bandwidth[client_idx]
                                    sum_tasks_select_idx += sorted_client_capacity[client_idx]
                                model_accuracy_select_idx = calculate_Γ(sum_accuracy_select_idx)
                                selection_possibilities_model_accuracies.append(model_accuracy_select_idx)
                                selection_possibilities_total_bandwidths.append(sum_bandwidth_select_idx)
                                selection_possibilities_num_tasks.append(sum_tasks_select_idx)
                            for u in range(len(selection_possibilities)):
                                # Check the constraints are whether satisfied.
                                if (selection_possibilities_model_accuracies[u] >= ε0 and
                                        selection_possibilities_total_bandwidths[u] <= B and
                                        selection_possibilities_num_tasks[u] <= num_tasks):
                                    # Calculate the total energy consumption of the qualified selection.
                                    total_energy_qualified_select_idx = 0
                                    for client_idx in selection_possibilities[u]:
                                        total_energy_qualified_select_idx += sorted_client_energy[client_idx]
                                    # Calculate the objective function.
                                    if selection_possibilities_model_accuracies[u] > 0:
                                        f_obj = (total_energy_qualified_select_idx /
                                                 selection_possibilities_model_accuracies[u])
                                    else:
                                        f_obj = inf
                                    obj = list(obj)
                                    # Store the objective function value.
                                    obj.append(f_obj)
                                    # Store the qualified selection.
                                    qualified_selection.append(selection_possibilities[u])
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
                                        client_index = sorted_client_index[qs_idx]
                                        if num_tasks_assigned < num_tasks:
                                            if beta_star_tasks_i[client_index] + sorted_client_capacity[qs_idx] \
                                                    <= init_qualified_client_max_capacity[client_index]:
                                                tasks_to_assign = sorted_client_capacity[qs_idx]
                                                if num_tasks_assigned + tasks_to_assign > num_tasks:
                                                    tasks_to_assign = num_tasks - num_tasks_assigned
                                                beta_star_i[client_index] = 1
                                                beta_star_tasks_i[client_index] += tasks_to_assign
                                                num_tasks_assigned += tasks_to_assign
                                            else:
                                                tasks_to_assign = (init_qualified_client_max_capacity[client_index]
                                                                   - beta_star_tasks_i[client_index])
                                                if num_tasks_assigned + tasks_to_assign > num_tasks:
                                                    tasks_to_assign = num_tasks - num_tasks_assigned
                                                beta_star_tasks_i[client_index] += tasks_to_assign
                                                num_tasks_assigned += tasks_to_assign
                                else:
                                    f_obj_beta_star_i = sorted_client_n[m]
                                    beta_star_i = the_first_qualified_client_index
                                    beta_star_tasks_i = the_first_qualified_client_index_tasks
                            else:
                                beta_star_i = the_first_qualified_client_index
                                beta_star_tasks_i = the_first_qualified_client_index_tasks
                            break
                        else:
                            m = m + 1
        # Organizing the solution for the round i.
        selected_clients_i = []
        makespan_i = 0
        energy_consumption_i = 0
        sum_accuracy_i = 0
        for client_index, _ in enumerate(beta_star_i):
            if beta_star_i[client_index] == 1:
                selected_clients_i.append(client_index)
                selected_client_num_tasks = beta_star_tasks_i[client_index]
                makespan_ik = Ti[client_index][selected_client_num_tasks-1]
                if makespan_ik > makespan_i:
                    makespan_i = makespan_ik
                energy_consumption_ik = Ei[client_index][selected_client_num_tasks-1]
                energy_consumption_i += energy_consumption_ik
                accuracy_ik = εi[client_index][selected_client_num_tasks-1]
                sum_accuracy_i += accuracy_ik
        beta_star.append(beta_star_i)
        beta_star_tasks.append(beta_star_tasks_i)
        f_obj_beta_star.append(f_obj_beta_star_i)
        selected_clients.append(selected_clients_i)
        makespan.append(makespan_i)
        energy_consumption.append(energy_consumption_i)
        accuracy_i = calculate_Γ(sum_accuracy_i)
        training_accuracy.append(accuracy_i)
    return (beta_star, beta_star_tasks, f_obj_beta_star, selected_clients, makespan, energy_consumption,
            training_accuracy)
