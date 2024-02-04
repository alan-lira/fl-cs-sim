from numpy import log, ndarray
from numpy.random import seed, uniform


def create_logn_costs(rng_seed: int,
                      matrix: ndarray,
                      index: int,
                      tau: int,
                      verbose: bool,
                      low_random: int,
                      high_random: int) -> None:
    """
    Fills a row with the Cost matrix based on a log n function.

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
    low_random : int
        Lower limit of the uniform distribution used for sampling
    high_random : int
        Upper limit of the uniform distribution used for sampling
    Notes
    -----
    The log n function is on the format f(x) = a + b log(1+x),
    where a and b are randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    seed(rng_seed)
    # Generates alpha, beta and gamma
    alpha, beta = uniform(low_random, high_random, 2)
    if verbose:
        print("[Index: {0} | Seed: {1} | α: {2} | β: {3}] - "
              "Creating logn costs with f(x) = {2} + {3} * log(x + 1), for x in [0, {4}]."
              .format(index, rng_seed, alpha, beta, tau + 1))
    # Fills row in the matrix
    matrix[index][:] = [(alpha + (beta * log(x + 1))) for x in range(tau + 1)]
