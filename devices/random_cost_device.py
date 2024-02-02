from numpy import ndarray
from numpy.random import seed, uniform


def create_random_costs(rng_seed: int,
                        matrix: ndarray,
                        index: int,
                        tau: int,
                        verbose: bool) -> None:
    """
    Fills a row with the Cost matrix based on a values from a uniform distribution.

    Parameters
    ----------
    rng_seed : int
        Seed to the random number generator
    matrix : np.ndarray
        Matrix of costs
    index : int
        Row of the matrix to fill
    tau : int
        Size of the row to fill (number of tasks)
    verbose : boolean (default False)
        True if information of the device should be printed
    Notes
    -----
    Values are randomly samples from a uniform
    distribution in the interval [0, tau).
    """
    # Sets RNG seed
    seed(rng_seed)
    # Generates random values
    values = uniform(0, tau, tau + 1)
    if verbose:
        print("[Index: {0} | Seed: {1}] - "
              "Creating random costs within the interval [0, {2})."
              .format(index, rng_seed, tau))
    # Fills row in the matrix
    matrix[index][:] = values
