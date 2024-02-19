from numpy import ndarray
from numpy.random import seed, uniform


def create_linear_costs(rng_seed: int,
                        matrix: ndarray,
                        index: int,
                        tau: int,
                        verbose: bool,
                        low_random: int,
                        high_random: int) -> None:
    """
    Fills a row with the Cost matrix based on a linear function.

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
    The linear function is on the format f(x) = a + bx,
    where a and b are randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    seed(rng_seed)
    # Generates alpha and beta
    alpha, beta = uniform(low_random, high_random, 2)
    if verbose:
        print("[Index: {0} | Seed: {1} | α: {2} | β: {3}] - "
              "Creating linear costs with f(x) = {2} + {3} * x, for x in [0, {4}]."
              .format(index, rng_seed, alpha, beta, tau))
    # Fills row in the matrix
    matrix[index][:] = [(alpha + (beta * x)) for x in range(tau + 1)]
