from math import log2, pow
from numpy import array, ndarray, ones

"""
Yu, L., Albelaihi, R., Sun, X., et al.: Jointly Optimizing Client Selection and Re-
source Management in Wireless Federated Learning for Internet of Things. IEEE
Internet of Things Journal 9(6), 4385–4395 (2021)
"""


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


def generate_t_wait_array(J: ndarray,
                          t_comp: ndarray,
                          t_up: ndarray) -> ndarray:
    t_wait = []
    for index, _ in enumerate(J):
        if index == 0:
            t_wait_j_plus_one = 0.0
            t_wait.append((J[index], t_wait_j_plus_one))
        else:
            t_comp_j = float(t_comp[index-1])
            t_wait_j = t_wait[index-1][1]
            t_up_j = float(t_up[index-1])
            t_comp_j_plus_one = float(t_comp[index])
            t_wait_j_plus_one = calculate_t_wait(t_comp_j, t_wait_j, t_up_j, t_comp_j_plus_one)
            t_wait.append((J[index], t_wait_j_plus_one))
    return array(t_wait)


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
                                       g: ndarray,
                                       D: ndarray,
                                       C: ndarray,
                                       f_max: ndarray,
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
    idx = []
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
        # Set idx.
        idx.append(i)
        # Set fi.
        fi = f_max[i]
        f.append(fi)
        # Set pi.
        pi = p_max
        p.append(pi)
        # Calculate t_comp_i.
        t_comp_i = calculate_t_comp(θ, ε, float(C[i]), float(D[i]), float(fi))
        t_comp.append(t_comp_i)
        # Calculate ri.
        ri = calculate_ri(B, pi, float(g[i]), N0)
        r.append(ri)
        # Calculate t_up_i.
        t_up_i = calculate_t_up(s, ri)
        t_up.append(t_up_i)
        # Calculate E_comp_i.
        E_comp_i = calculate_E_comp(t_comp_i, float(fi), γ)
        E_comp.append(E_comp_i)
        # Calculate E_up_i.
        E_up_i = calculate_E_up(pi, t_up_i)
        E_up.append(E_up_i)
        # Calculate ni.
        ni = calculate_ni(α, E_comp_i, E_up_i)
        n.append(ni)
    # Sort all the clients in increasing order based on ηi.
    # Denote I′ as the set of sorted clients.
    sorted_n, sorted_idx = map(list, zip(*sorted(zip(n, idx), reverse=False)))
    # Initialize x.
    x = ones(shape=(len(sorted_idx)), dtype=int)
    for _ in enumerate(sorted_idx):
        # Update the set of participants J based on Constraints (13) and (14).
        # Constraints (13) and (14) define the set of selected clients, which are sorted based on the
        # increasing order of their computational latency.
        # Constraint 13: J = {i ∈ I| xi = 1}
        # Constraint 14: ∀j ∈ J, t_comp_j ≤ t_comp_j+1
        idx_j = []
        t_j = []
        for index, _ in enumerate(x):
            if x[index] == 1:
                idxj = idx[index]
                idx_j.append(idxj)
                t_comp_j = t_comp[index]
                t_j.append(t_comp_j)
        sorted_t_j, sorted_J = map(list, zip(*sorted(zip(t_j, idx_j), reverse=False)))
        t_comp_aux = []
        t_up_aux = []
        for index, _ in enumerate(sorted_J):
            j_sorted_idx = sorted_J[index]
            t_comp_aux.append(t_comp[j_sorted_idx])
            t_up_aux.append(t_up[j_sorted_idx])
        t_comp_aux = array(t_comp_aux)
        t_up_aux = array(t_up_aux)
        # Calculate t_wait_j based on Eq. (6).
        t_wait_aux = generate_t_wait_array(sorted_J, t_comp_aux, t_up_aux)
        for index, _ in enumerate(sorted_J):
            idxj = idx_j[index]
            # Get t_comp_j, t_wait_j, and t_up_j, calculated based on Eqs. (3), (6), and (5), respectively.
            t_comp_j = t_comp_aux[index]
            t_wait_j = t_wait_aux[index][1]
            t_up_j = t_up_aux[index]
            if t_comp_j + t_wait_j + t_up_j > τ:
                x[idxj] = 0
                break
    # Organizing the solution.
    selected_clients = []
    makespan = 0
    energy_consumption = 0
    for index, _ in enumerate(x):
        if x[index] == 1:
            j = index  # Display the selected clients in ascending order.
            # j = sorted_idx[index]  # Display the selected clients sorted by n.
            t_wait_index = [i for i, x in enumerate(t_wait_aux) if x[0] == j][0]
            selected_clients.append(j)
            makespan_j = t_comp[j] + t_wait_aux[t_wait_index][1] + t_up[j]
            if makespan_j > makespan:
                makespan = makespan_j
            energy_consumption_j = E_comp[j] + E_up[j]
            energy_consumption += energy_consumption_j
    return x, selected_clients, makespan, energy_consumption
