from heapq import heapify, heappop, heappush
from numpy import array, copy, ndarray, sum


def olar_adapted(tasks: int,
                 resources: int,
                 cost: ndarray,
                 assignment_capacities: ndarray) -> ndarray:
    """
    Finds an assignment of tasks to resources using OLAR.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    assignment_capacities : ndarray(shape=(resources))
        Task assignment capacities per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    heap = []
    lower_limits = array([min(assignment_capacity_i) for assignment_capacity_i in assignment_capacities])
    upper_limits = array([max(assignment_capacity_i) for assignment_capacity_i in assignment_capacities])
    # Assigns lower limit to all resources
    assignment = copy(lower_limits)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limits[i]:
            heap.append((cost[i][assignment[i]+1], i))
    heapify(heap)
    # Computes zeta (sum of lower limits)
    zeta = sum(lower_limits)
    # Iterates assigning the remaining tasks
    for t in range(zeta+1, tasks+1):
        c, j = heappop(heap)  # Find minimum cost
        assignment[j] += 1  # Assigns task t
        # Checks if more tasks can be assigned to j
        if assignment[j] < upper_limits[j]:
            heappush(heap, (cost[j][assignment[j]+1], j))
    return assignment
