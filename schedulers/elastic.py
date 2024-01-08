from math import log2, pow
from numpy import array, ones


def calculate_t_comp(θ: float,
                     ε: float,
                     Ci: float,
                     Di: float,
                     fi: float) -> float:
    return θ * log2(1/ε) * ((Ci * Di) / fi)


def calculate_t_wait(t_comp_j: float,
                     t_wait_j: float,
                     t_up_j: float,
                     t_comp_j_plus_one: float) -> float:
    return max(0.0, t_comp_j + t_wait_j + t_up_j - t_comp_j_plus_one)


def generate_t_wait_array(J: array,
                          t_comp: array,
                          t_up: array) -> array:
    t_wait = []
    for index, _ in enumerate(J):
        if index == 0:
            t_wait_j_plus_one = 0.0
            t_wait.append((J[index], t_wait_j_plus_one))
        else:
            t_comp_j = t_comp[index-1]
            t_wait_j = t_wait[index-1][1]
            t_up_j = t_up[index-1]
            t_comp_j_plus_one = t_comp[index]
            t_wait_j_plus_one = calculate_t_wait(t_comp_j, t_wait_j, t_up_j, t_comp_j_plus_one)
            t_wait.append((J[index], t_wait_j_plus_one))
    return t_wait


def calculate_ri(B: float,
                 pi: float,
                 gi: float,
                 N0: float) -> float:
    return B * log2(1 + ((pi * gi) / N0 * B))


def calculate_t_up(s: float,
                   ri: float) -> float:
    return s / ri


def calculate_E_comp(t_comp_i: float,
                     fi: float,
                     γ: float) -> float:
    return t_comp_i * fi * γ * pow(fi, 2)


def calculate_E_up(pi: float,
                   t_up_i: float) -> float:
    return pi * t_up_i


def calculate_ni(α: float,
                 E_comp_i: float,
                 E_up_i: float) -> float:
    return α * (E_comp_i + E_up_i + 1) - 1


def elastic_client_selection_algorithm(I: int,
                                       g: array,
                                       D: array,
                                       C: array,
                                       f_max: array,
                                       p_max: float,
                                       N0: float,
                                       B: float,
                                       s: float,
                                       θ: float,
                                       ε: float,
                                       τ: float,
                                       γ: float,
                                       α: float) -> tuple:
    # ∀i ∈ I, fi = f_max_i , pi = p_max_i , and compute ηi.
    t_comp = []
    t_up = []
    t_wait_aux = []
    E_comp = []
    E_up = []
    f = []
    p = []
    r = []
    n = []
    for i in range(I):
        # Set fi.
        fi = f_max[i]
        f.append(fi)
        # Set pi.
        pi = p_max
        p.append(pi)
        # Calculate t_comp_i.
        t_comp_i = calculate_t_comp(θ, ε, C[i], D[i], fi)
        t_comp.append(t_comp_i)
        # Calculate ri.
        ri = calculate_ri(B, pi, g[i], N0)
        r.append(ri)
        # Calculate t_up_i.
        t_up_i = calculate_t_up(s, ri)
        t_up.append(t_up_i)
        # Calculate E_comp_i.
        E_comp_i = calculate_E_comp(t_comp_i, fi, γ)
        E_comp.append(E_comp_i)
        # Calculate E_up_i.
        E_up_i = calculate_E_up(pi, t_up_i)
        E_up.append(E_up_i)
        # Calculate ni.
        ni = calculate_ni(α, E_comp_i, E_up_i)
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
        # Constraint 14: ∀j ∈ J, t_comp_j ≤ t_comp_j+1
        J = []
        for index, _ in enumerate(x):
            if x[index] == 1:
                j = I_line_sorted[index]
                t_comp_j = t_comp[index]
                J.append([j, t_comp_j])
        J_sorted = sorted(J, key=lambda x: x[1])
        J_sorted = [x[0] for x in J_sorted]
        J = J_sorted
        t_comp_aux = []
        t_up_aux = []
        for index, _ in enumerate(J):
            j = J[index]
            t_comp_aux.append(t_comp[j])
            t_up_aux.append(t_up[j])
        t_comp_aux = array(t_comp_aux)
        t_up_aux = array(t_up_aux)
        # Calculate t_wait_j based on Eq. (6).
        t_wait_aux = generate_t_wait_array(J, t_comp_aux, t_up_aux)
        for index, _ in enumerate(J):
            # Get t_comp_j, t_wait_j, and t_up_j, calculated based on Eqs. (3), (6), and (5), respectively.
            t_comp_j = t_comp_aux[index]
            t_wait_j = t_wait_aux[index][1]
            t_up_j = t_up_aux[index]
            if t_comp_j + t_wait_j + t_up_j > τ:
                x[i] = 0
                break
    # Organizing the solution.
    selected_clients = []
    makespan = 0
    energy_consumption = 0
    for index, _ in enumerate(x):
        if x[index] == 1:
            j = I_line_sorted[index]
            t_wait_index = [i for i, x in enumerate(t_wait_aux) if x[0] == j][0]
            selected_clients.append(j)
            makespan_j = t_comp[j] + t_wait_aux[t_wait_index][1] + t_up[j]
            if makespan_j > makespan:
                makespan = makespan_j
            energy_consumption_j = E_comp[j] + E_up[j]
            energy_consumption += energy_consumption_j
    return x, selected_clients, makespan, energy_consumption
