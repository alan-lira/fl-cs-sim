from numpy import array, full, inf, ndarray, zeros


def min_max_time(num_resources: int,
                 num_tasks: int,
                 assignment_capacities: ndarray,
                 time_costs: ndarray) -> float:
    """
    First step of MEC With Accuracy: finds the minimal makespan (Cₘₐₓ) to assign tasks to resources.
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
    Returns
    -------
    minimal_makespan : float
        Minimal makespan (Cₘₐₓ)
    """
    # (I) Filtering: nothing to do.
    # (II) Initialization: minimal costs and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_time_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities[0]:
        partial_solutions[0][j] = j
        minimal_time_costs[0][j] = time_costs[0][j]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities[i]:
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                time_cost_new_solution = max(float(minimal_time_costs[i-1][t-j]), time_costs[i][j])
                if time_cost_new_solution < minimal_time_costs[i][t]:
                    # New best solution for Zᵢ(t).
                    minimal_time_costs[i][t] = time_cost_new_solution
                    partial_solutions[i][t] = j
    # (V) Organize the final solution.
    minimal_makespan = float(minimal_time_costs[num_resources-1][num_tasks])
    # Return the minimal makespan.
    return minimal_makespan


def min_sum_energy_then_max_sum_accuracy(num_resources: int,
                                         num_tasks: int,
                                         assignment_capacities: ndarray,
                                         time_costs: ndarray,
                                         energy_costs: ndarray,
                                         training_accuracies: ndarray,
                                         max_makespan: float) -> tuple:
    """
    Second step of MEC With Accuracy: finds the minimal energy consumption (ΣE) and then
    the maximal training accuracy (ΣW), in order, to assign tasks to resources, while respecting the time limit (C).
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
    training_accuracies : ndarray(shape=(num_resources, num_tasks+1), object)
        Training accuracies obtained when processing tasks per resource (W)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    max_makespan : float
        Time limit (C)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_energy_consumption : float, maximal_training_accuracy: float
        Optimal schedule (X*), minimal energy consumption (ΣE), and maximal training accuracy (ΣW).
    """
    # (I) Filtering: only assignments that respect the maximum makespan.
    for i in range(0, num_resources):
        assignment_capacities_i = []
        for j in assignment_capacities[i]:
            if time_costs[i][j] <= max_makespan:
                assignment_capacities_i.append(j)
        assignment_capacities[i] = array(assignment_capacities_i)
    # (II) Initialization: minimal energy costs, maximal training accuracies, and partial solutions matrices.
    partial_solutions = zeros(shape=(num_resources, num_tasks+1), dtype=int)
    minimal_energy_costs = full(shape=(num_resources, num_tasks+1), fill_value=inf, dtype=float)
    maximal_training_accuracies = zeros(shape=(num_resources, num_tasks+1), dtype=float)
    # (III) Solutions for the first resource (Z₁).
    for j in assignment_capacities[0]:
        partial_solutions[0][j] = j
        minimal_energy_costs[0][j] = energy_costs[0][j]
        maximal_training_accuracies[0][j] = training_accuracies[0][j]
    # Solutions for other resources (Zᵢ).
    for i in range(1, num_resources):
        # Test all assignments to resource i.
        for j in assignment_capacities[i]:
            for t in range(j, num_tasks+1):
                # (IV) Test new solution.
                energy_cost_new_solution = minimal_energy_costs[i-1][t-j] + energy_costs[i][j]
                training_accuracy_new_solution = maximal_training_accuracies[i-1][t-j] + training_accuracies[i][j]
                if energy_cost_new_solution < minimal_energy_costs[i][t] or \
                    (energy_cost_new_solution == minimal_energy_costs[i][t] and
                     training_accuracy_new_solution > maximal_training_accuracies[i][t]):
                    # New best solution for Zᵢ(t).
                    minimal_energy_costs[i][t] = energy_cost_new_solution
                    maximal_training_accuracies[i][t] = training_accuracy_new_solution
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
    maximal_training_accuracy = maximal_training_accuracies[num_resources-1][num_tasks]
    # Return the optimal schedule, the minimal energy consumption, and the maximal training accuracy.
    return optimal_schedule, minimal_energy_consumption, maximal_training_accuracy


def mec_with_accuracy(num_resources: int,
                      num_tasks: int,
                      assignment_capacities: ndarray,
                      time_costs: ndarray,
                      energy_costs: ndarray,
                      training_accuracies: ndarray) -> tuple:
    """
    Minimal Makespan, Minimal Energy Consumption, and Maximal Training Accuracy FL Schedule problem (MEC With Accuracy):
    finds an optimal schedule (X*) that minimizes the makespan (Cₘₐₓ), minimizes the total energy consumption (ΣE),
    and maximizes the training accuracy (ΣW), in order.
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
    training_accuracies : ndarray(shape=(num_resources, num_tasks+1), object)
        Training accuracies obtained when processing tasks per resource (W)
    energy_costs : ndarray(shape=(num_resources, num_tasks+1), object)
        Energy costs to process tasks per resource (ε)
    Returns
    -------
    optimal_schedule : ndarray(shape=(num_resources), int), minimal_makespan : float, minimal_energy_consumption : float, maximal_training_accuracy : float
        Optimal schedule (X*), minimal makespan (Cₘₐₓ), minimal energy consumption (ΣE), and maximal training accuracy (ΣW)
    """
    # First step: compute the MinMaxTime algorithm (minimal makespan optimization).
    minimal_makespan = min_max_time(num_resources,
                                    num_tasks,
                                    assignment_capacities,
                                    time_costs)
    # Second step: compute the MinSumEnergyThenMaxSumAccuracy algorithm (minimal energy consumption optimization).
    optimal_schedule, minimal_energy_consumption, maximal_training_accuracy \
        = min_sum_energy_then_max_sum_accuracy(num_resources,
                                               num_tasks,
                                               assignment_capacities,
                                               time_costs,
                                               energy_costs,
                                               training_accuracies,
                                               minimal_makespan)
    # Return the optimal schedule, the minimal makespan, the minimal energy consumption,
    # and the maximal training accuracy.
    return optimal_schedule, minimal_makespan, minimal_energy_consumption, maximal_training_accuracy
