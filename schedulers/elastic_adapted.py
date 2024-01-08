from numpy import array, ones


def calculate_ni_adapted(α: float,
                         Ei: float) -> float:
    return α * (Ei + 1) - 1


def elastic_adapted_client_selection_algorithm(I: int,
                                               A: array,
                                               t: array,
                                               E: array,
                                               τ: float,
                                               α: float) -> tuple:
    # Some remarks about this adapted version of ELASTIC algorithm:
    # 1. We considered that clients does not share a wireless channel, so they can upload their model
    #    without having to wait for the channel availability. In other words, ∀i ∈ I, t_wait_i = 0.
    # 2. The algorithm receives a previously generated array of task assignment capacities (A),
    #    such that the client i can process exactly Ai tasks.
    # 3. The algorithm receives a previously generated array of time costs (t), such that ti = t_comp_i + t_up_i.
    # 4. The algorithm receives a previously generated array of energy costs (E), such that Ei = E_comp_i + E_up_i.

    # ∀i ∈ I, compute ηi.
    n = []
    for i in range(I):
        ni = calculate_ni_adapted(α, E[i][A[i]-1])
        n.append(ni)
    # Sort all the clients in increasing order based on ηi.
    # Denote I′ as the set of sorted clients.
    I_line_sorted = []
    for i in range(I):
        I_line_sorted.append([i, n[i]])
    I_line_sorted = sorted(I_line_sorted, key=lambda x: x[1])
    I_line_sorted = [x[0] for x in I_line_sorted]
    # Initialize x.
    x = ones(shape=(len(I_line_sorted)), dtype=int)
    for i in I_line_sorted:
        # Update the set of participants J based on Constraints (13) and (14).
        # Constraints (13) and (14) define the set of selected clients, which are sorted based on the
        # increasing order of their computational latency.
        # Constraint 13: J = {i ∈ I| xi = 1}
        # Constraint 14: ∀j ∈ J, t_comp_j ≤ t_comp_j+1 (Considering this adaptation, t_j ≤ t_j+1)
        J = []
        for index, _ in enumerate(x):
            if x[index] == 1:
                j = I_line_sorted[index]
                t_j = t[index][A[index]-1]
                J.append([j, t_j])
        J_sorted = sorted(J, key=lambda x: x[1])
        J_sorted = [x[0] for x in J_sorted]
        J = J_sorted
        for index, _ in enumerate(J):
            j = J[index]
            if t[j][A[j]-1] > τ:
                x[i] = 0
                break
    # Organize the solution.
    selected_clients = []
    makespan = 0
    energy_consumption = 0
    for index, _ in enumerate(x):
        if x[index] == 1:
            j = I_line_sorted[index]
            selected_clients.append(j)
            makespan_j = t[j][A[j]-1]
            if makespan_j > makespan:
                makespan = makespan_j
            energy_consumption_j = E[j][A[j]-1]
            energy_consumption += energy_consumption_j
    return x, selected_clients, makespan, energy_consumption
