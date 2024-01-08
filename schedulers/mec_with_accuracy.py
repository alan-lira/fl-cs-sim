from numpy import array, full, inf, ndarray, zeros


def min_max_time_with_accuracy(num_resources: int,
                               num_tasks: int,
                               assignment_capacities: ndarray,
                               time_costs: ndarray,
                               accuracy_gains: ndarray) -> tuple:
    """
    First step of MEC With Accuracy: finds the minimal makespan (Cₘₐₓ) and the maximal weighted accuracy (Accₘₐₓ),
    in order, to assign tasks to resources.
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    accuracy_gains : ndarray(shape=(num_resources, num_tasks+1), object)
        Accuracy gains to process tasks per resource (W)
    Returns
    -------
    minimal_makespan : float, maximal_weighted_accuracy : float
        Minimal makespan (Cₘₐₓ) and maximal weighted accuracy (Accₘₐₓ)
    """
    # (I) Filtering: nothing to do.
    # (II) Initialization: minimal costs, maximal gains and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_time_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    maximal_weighted_accuracy_gains = zeros(shape=(num_resources, num_tasks+1), dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities[0]:
        partial_solutions[0][j] = j
        minimal_time_costs[0][j] = time_costs[0][j]
        maximal_weighted_accuracy_gains[0][j] = accuracy_gains[0][j] / j
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities[i]:
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                time_cost_new_solution = max(float(minimal_time_costs[i-1][t-j]), time_costs[i][j])
                weighted_accuracy_new_solution = maximal_weighted_accuracy_gains[i-1][t-j] + (accuracy_gains[i][j] / j)
                if (time_cost_new_solution < minimal_time_costs[i][t] or
                   (time_cost_new_solution == minimal_time_costs[i][t] and
                   weighted_accuracy_new_solution > maximal_weighted_accuracy_gains[i][t])):
                    # New best solution for Zᵢ(t).
                    minimal_time_costs[i][t] = time_cost_new_solution
                    maximal_weighted_accuracy_gains[i][t] = weighted_accuracy_new_solution
                    partial_solutions[i][t] = j
    # (V) Organize the final solution.
    minimal_makespan = float(minimal_time_costs[num_resources-1][num_tasks])
    maximal_weighted_accuracy = float(maximal_weighted_accuracy_gains[num_resources-1][num_tasks])
    # Return the minimal makespan and the maximal weighted accuracy.
    return minimal_makespan, maximal_weighted_accuracy


def min_sum_energy_with_accuracy(num_resources: int,
                                 num_tasks: int,
                                 assignment_capacities: ndarray,
                                 time_costs: ndarray,
                                 accuracy_gains: ndarray,
                                 energy_costs: ndarray,
                                 max_makespan: float,
                                 min_weighted_accuracy: float) -> tuple:
    """
    Second step of MEC With Accuracy: finds the minimal energy consumption (ΣE) to assign tasks to resources,
    while respecting the time limit (C) and the minimum weighted accuracy (Accₘᵢₙ).
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    accuracy_gains : ndarray(shape=(num_resources, num_tasks+1), object)
        Accuracy gains to process tasks per resource (W)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    max_makespan : float
        Time limit (C)
    min_weighted_accuracy : float
        Minimum weighted accuracy (Accₘᵢₙ)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_energy_consumption : float
        Optimal schedule (X*) and minimal energy consumption (ΣE)
    """
    # (I) Filtering: only assignments that respect the maximum makespan.
    for i in range(0, num_resources):
        assignment_capacities_i = []
        for j in assignment_capacities[i]:
            if (time_costs[i][j] <= max_makespan and
               ((accuracy_gains[i][j] / j) >= ((min_weighted_accuracy / num_resources) / (num_tasks - (j-1))))):
                assignment_capacities_i.append(j)
        assignment_capacities[i] = array(assignment_capacities_i)
    # (II) Initialization: minimal costs and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_energy_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities[0]:
        partial_solutions[0][j] = j
        minimal_energy_costs[0][j] = energy_costs[0][j]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities[i]:
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                energy_cost_new_solution = minimal_energy_costs[i-1][t-j] + energy_costs[i][j]
                if energy_cost_new_solution < minimal_energy_costs[i][t]:
                    # New best solution for Zᵢ(t).
                    minimal_energy_costs[i][t] = energy_cost_new_solution
                    partial_solutions[i][t] = j
    # Extract the optimal schedule.
    t = num_tasks
    optimal_schedule = zeros(num_resources, dtype=int)
    for i in reversed(range(num_resources)):
        j = partial_solutions[i][t]  # Number of tasks to assign to resource i.
        optimal_schedule[i] = j
        t = t-j  # Solution index of resource i-1.
    # (V) Organize the final solution.
    minimal_energy_consumption = minimal_energy_costs[num_resources-1][num_tasks]
    # Return the optimal schedule and the minimal energy consumption.
    return optimal_schedule, minimal_energy_consumption


def mec_with_accuracy(num_resources: int,
                      num_tasks: int,
                      assignment_capacities: ndarray,
                      time_costs: ndarray,
                      accuracy_gains: ndarray,
                      energy_costs: ndarray) -> tuple:
    """
    Minimal Makespan, Maximal Weighted Accuracy, and Minimal Energy Consumption FL Schedule problem (MEC With Accuracy):
    finds an optimal schedule (X*) that minimizes the makespan (Cₘₐₓ), maximizes the weighted accuracy (Accₘₐₓ),
    and minimizes the total energy consumption (ΣE), in order.
    Parameters
    ----------
    num_resources : int
        Number of resources (n)
    num_tasks : int
        Number of tasks (T)
    assignment_capacities : ndarray(shape=(num_resources), int)
        Task assignment capacities per resource (A)
    time_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Time costs to process tasks per resource (Ρ)
    accuracy_gains : ndarray(shape=(num_resources, num_tasks+1), object)
        Accuracy gains to process tasks per resource (W)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_makespan : float, maximal_weighted_accuracy : float, minimal_energy_consumption : float
        Optimal schedule (X*), minimal makespan (Cₘₐₓ), maximal weighted accuracy (Accₘₐₓ), and minimal energy consumption (ΣE)
    """
    # First step: compute the MinMaxTimeWithAccuracy algorithm (minimal makespan optimization).
    minimal_makespan, maximal_weighted_accuracy = min_max_time_with_accuracy(num_resources,
                                                                             num_tasks,
                                                                             assignment_capacities,
                                                                             time_costs,
                                                                             accuracy_gains)
    # Second step: compute the MinSumEnergyWithAccuracy algorithm (minimal energy consumption optimization).
    optimal_schedule, minimal_energy_consumption = min_sum_energy_with_accuracy(num_resources,
                                                                                num_tasks,
                                                                                assignment_capacities,
                                                                                time_costs,
                                                                                accuracy_gains,
                                                                                energy_costs,
                                                                                minimal_makespan,
                                                                                maximal_weighted_accuracy)
    # Return the optimal schedule, the minimal makespan, the maximal weighted accuracy,
    # and the minimal energy consumption.
    return optimal_schedule, minimal_makespan, maximal_weighted_accuracy, minimal_energy_consumption
