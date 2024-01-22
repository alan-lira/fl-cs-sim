from numpy import cumsum, ndarray
from numpy.random import seed, uniform


def create_recursive_costs(rng_seed: int,
                           matrix: ndarray,
                           index: int,
                           tau: int,
                           verbose: bool,
                           low_random: int,
                           high_random: int) -> None:
    """
    Fills a row with the Cost matrix based on a recursive function.

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
    The recursive function is on the format f(x) = f(x-1) + a,
    where a is randomly sampled from a uniform
    distribution in the interval [low_random, high_random).
    """
    # Sets RNG seed
    seed(rng_seed)
    # Generates random values
    values = uniform(low_random, high_random, tau+1)
    if verbose:
        print(f'[{index}] - Creating random costs with f(x) =' +
              ' f(x-1) + alpha' +
              f'. RNG seed = {rng_seed}')
    # Fills row in the matrix
    matrix[index][:] = cumsum(values)
