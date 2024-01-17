from heapq import heapify, heappop, heappush
from numpy import copy, ndarray, sum


def olar(tasks: int,
         resources: int,
         cost: ndarray,
         lower_limit: ndarray,
         upper_limit: ndarray) -> ndarray:
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
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    heap = []
    # Assigns lower limit to all resources
    assignment = copy(lower_limit)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limit[i]:
            heap.append((cost[i][assignment[i]+1], i))
    heapify(heap)
    # Computes zeta (sum of lower limits)
    zeta = sum(lower_limit)
    # Iterates assigning the remaining tasks
    for t in range(zeta+1, tasks+1):
        c, j = heappop(heap)  # Find minimum cost
        assignment[j] += 1  # Assigns task t
        # Checks if more tasks can be assigned to j
        if assignment[j] < upper_limit[j]:
            heappush(heap, (cost[j][assignment[j]+1], j))
    return assignment
