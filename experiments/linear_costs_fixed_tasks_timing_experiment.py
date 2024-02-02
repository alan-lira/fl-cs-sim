"""
# Description of the Experiment:
#  - We generate the costs to up to 2000 tasks for 10 to 100 resources with steps of 15.
#  - All resources costs follow linear functions with RNG seeds starting at 100.
#  - Time costs were generated using uniformly distributed values ranging from 1 to 10.
#  - Energy costs were generated using uniformly distributed values ranging from 0.32 to 3.2.
#  - Accuracy costs were generated using uniformly distributed values ranging from 0.2 to 2.0 (then normalized).
#  - We schedule 2000 tasks.
#  - We run adapted versions of OLAR; MC²MKP; ELASTIC; and FedAECS; and
#    MEC; MEC+Acc; ECMTC; and ECMTC+Acc schedulers.
#  - We use no lower or upper task assignment limits for resources.
#  - Each sample is composed of 5 executions of the schedulers.
#  - We get 20 samples for each pair (scheduler; tasks)
#  - The order of execution of the different schedulers is randomly defined.
#    We set shuffle seed at 1000 and increase it every time we need a new order.
#  - Every result is logged to a CSV file.
"""

from datetime import datetime
from numpy import arange, array, expand_dims, full, inf, zeros
from numpy.random import seed, shuffle
from pathlib import Path
from time import perf_counter
from timeit import timeit

from devices.linear_cost_device import create_linear_costs
from schedulers.ecmtc import ecmtc
from schedulers.ecmtc_plus_acc import ecmtc_plus_acc
from schedulers.elastic_adapted import elastic_adapted_client_selection_algorithm
from schedulers.fedaecs_adapted import fedaecs_adapted
from schedulers.mc2mkp_adapted import mc2mkp_adapted
from schedulers.mec import mec
from schedulers.mec_plus_acc import mec_plus_acc
from schedulers.olar_adapted import olar_adapted
from util.experiment_util import get_num_selected_resources, get_makespan, get_total_cost
from util.logger_util import Logger


def execute_scheduler(scheduler_execution_parameters: dict) -> dict:
    # Get the scheduler's execution parameters.
    index = scheduler_execution_parameters["index"]
    scheduler_name = scheduler_execution_parameters["scheduler_name"]
    num_tasks = scheduler_execution_parameters["num_tasks"]
    num_resources = scheduler_execution_parameters["num_resources"]
    time_costs = scheduler_execution_parameters["time_costs"]
    energy_costs = scheduler_execution_parameters["energy_costs"]
    training_accuracies = scheduler_execution_parameters["training_accuracies"]
    assignment_capacities = scheduler_execution_parameters["assignment_capacities"]
    scheduler_execution_result = None
    if scheduler_name == "OLAR":
        # Run the adapted version of OLAR algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        olar_adapted_assignment = olar_adapted(num_tasks,
                                               num_resources,
                                               time_costs,
                                               assignment_capacities)
        olar_adapted_num_selected_resources = get_num_selected_resources(olar_adapted_assignment)
        olar_adapted_makespan = get_makespan(time_costs, olar_adapted_assignment)
        olar_adapted_energy_consumption = get_total_cost(energy_costs, olar_adapted_assignment)
        olar_adapted_training_accuracy = get_total_cost(training_accuracies, olar_adapted_assignment)
        olar_adapted_result = {"scheduler_name": scheduler_name,
                               "num_tasks": num_tasks,
                               "num_resources": num_resources,
                               "assignment": olar_adapted_assignment,
                               "num_selected_resources": olar_adapted_num_selected_resources,
                               "makespan": olar_adapted_makespan,
                               "energy_consumption": olar_adapted_energy_consumption,
                               "training_accuracy": olar_adapted_training_accuracy}
        scheduler_execution_result = olar_adapted_result
    elif scheduler_name == "MC²MKP":
        # Run the adapted version of MC²MKP algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        mc2mkp_adapted_assignment = mc2mkp_adapted(num_tasks,
                                                   num_resources,
                                                   time_costs,
                                                   assignment_capacities)
        mc2mkp_adapted_num_selected_resources = get_num_selected_resources(mc2mkp_adapted_assignment)
        mc2mkp_adapted_makespan = get_makespan(time_costs, mc2mkp_adapted_assignment)
        mc2mkp_adapted_energy_consumption = get_total_cost(energy_costs, mc2mkp_adapted_assignment)
        mc2mkp_adapted_training_accuracy = get_total_cost(training_accuracies, mc2mkp_adapted_assignment)
        mc2mkp_adapted_result = {"scheduler_name": scheduler_name,
                                 "num_tasks": num_tasks,
                                 "num_resources": num_resources,
                                 "assignment": mc2mkp_adapted_assignment,
                                 "num_selected_resources": mc2mkp_adapted_num_selected_resources,
                                 "makespan": mc2mkp_adapted_makespan,
                                 "energy_consumption": mc2mkp_adapted_energy_consumption,
                                 "training_accuracy": mc2mkp_adapted_training_accuracy}
        scheduler_execution_result = mc2mkp_adapted_result
    elif scheduler_name == "ELASTIC":
        # Run the adapted version of ELASTIC algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        τ = inf  # Time limit in seconds (deadline).
        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.
        assignment_capacities_elastic = []
        # Divide the tasks as equally possible.
        mean_tasks = num_tasks // num_resources
        # But it still may have some leftovers. If so, they will be added to the first resource.
        leftover = num_tasks % num_resources
        for _ in range(num_resources):
            Ai = mean_tasks
            assignment_capacities_elastic.append(Ai)
        assignment_capacities_elastic[0] += leftover
        assignment_capacities_elastic = array(assignment_capacities_elastic)
        (elastic_adapted_assignment, elastic_adapted_tasks_assignment, _, elastic_adapted_makespan,
         elastic_adapted_energy_consumption) \
            = elastic_adapted_client_selection_algorithm(num_resources,
                                                         assignment_capacities_elastic,
                                                         time_costs,
                                                         energy_costs,
                                                         τ,
                                                         α)
        elastic_adapted_num_selected_resources = get_num_selected_resources(elastic_adapted_assignment)
        elastic_adapted_training_accuracy = get_total_cost(training_accuracies,
                                                           elastic_adapted_tasks_assignment)
        elastic_adapted_result = {"scheduler_name": scheduler_name,
                                  "num_tasks": num_tasks,
                                  "num_resources": num_resources,
                                  "assignment": elastic_adapted_tasks_assignment,
                                  "num_selected_resources": elastic_adapted_num_selected_resources,
                                  "makespan": elastic_adapted_makespan,
                                  "energy_consumption": elastic_adapted_energy_consumption,
                                  "training_accuracy": elastic_adapted_training_accuracy}
        scheduler_execution_result = elastic_adapted_result
    elif scheduler_name == "FedAECS":
        # Run the adapted version of FedAECS algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        num_rounds = 1  # Number of rounds.
        # Expanded shape of cost functions (FedAECS considers communication rounds).
        assignment_capacities_expanded_shape = expand_dims(assignment_capacities, axis=0)
        time_costs_expanded_shape = expand_dims(time_costs, axis=0)
        energy_costs_expanded_shape = expand_dims(energy_costs, axis=0)
        training_accuracies_expanded_shape = expand_dims(training_accuracies, axis=0)
        # Bandwidth information per resource per round.
        b_shape = (num_rounds,
                   num_resources,
                   len(assignment_capacities_expanded_shape[num_rounds-1][num_resources-1]))
        b = zeros(shape=b_shape)
        ε0 = 0.0  # The lower bound of accuracy.
        T_max = inf  # Deadline of a global iteration in seconds.
        B = inf  # Total bandwidth in Hz.
        (fedaecs_adapted_assignment, fedaecs_adapted_tasks_assignment, _, _, fedaecs_adapted_makespan,
         fedaecs_adapted_energy_consumption, fedaecs_adapted_training_accuracy) \
            = fedaecs_adapted(num_rounds,
                              num_resources,
                              num_tasks,
                              assignment_capacities_expanded_shape,
                              time_costs_expanded_shape,
                              energy_costs_expanded_shape,
                              training_accuracies_expanded_shape,
                              b,
                              ε0,
                              T_max,
                              B)
        fedaecs_adapted_num_selected_resources = get_num_selected_resources(fedaecs_adapted_assignment[0])
        # fedaecs_adapted_makespan = get_makespan(time_costs, fedaecs_adapted_assignment[0])
        # fedaecs_adapted_energy_consumption = get_total_cost(energy_costs, fedaecs_adapted_assignment[0])
        # fedaecs_adapted_training_accuracy = get_total_cost(training_accuracies, fedaecs_adapted_assignment[0])
        fedaecs_adapted_result = {"scheduler_name": scheduler_name,
                                  "num_tasks": num_tasks,
                                  "num_resources": num_resources,
                                  "assignment": fedaecs_adapted_tasks_assignment[0],
                                  "num_selected_resources": fedaecs_adapted_num_selected_resources,
                                  "makespan": fedaecs_adapted_makespan[0],
                                  "energy_consumption": fedaecs_adapted_energy_consumption[0],
                                  "training_accuracy": fedaecs_adapted_training_accuracy[0]}
        scheduler_execution_result = fedaecs_adapted_result
    elif scheduler_name == "MEC":
        # Run the MEC algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        mec_assignment, mec_makespan, mec_energy_consumption \
            = mec(num_resources,
                  num_tasks,
                  assignment_capacities,
                  time_costs,
                  energy_costs)
        mec_num_selected_resources = get_num_selected_resources(mec_assignment)
        # mec_makespan = get_makespan(time_costs, mec_assignment)
        # mec_energy_consumption = get_total_cost(energy_costs, mec_assignment)
        mec_training_accuracy = get_total_cost(training_accuracies, mec_assignment)
        mec_result = {"scheduler_name": scheduler_name,
                      "num_tasks": num_tasks,
                      "num_resources": num_resources,
                      "assignment": mec_assignment,
                      "num_selected_resources": mec_num_selected_resources,
                      "makespan": mec_makespan,
                      "energy_consumption": mec_energy_consumption,
                      "training_accuracy": mec_training_accuracy}
        scheduler_execution_result = mec_result
    elif scheduler_name == "MEC+Acc":
        # Run the MEC+Acc algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        (mec_plus_acc_assignment, mec_plus_acc_makespan, mec_plus_acc_energy_consumption,
         mec_plus_acc_training_accuracy) \
            = mec_plus_acc(num_resources,
                           num_tasks,
                           assignment_capacities,
                           time_costs,
                           energy_costs,
                           training_accuracies)
        mec_plus_acc_num_selected_resources = get_num_selected_resources(mec_plus_acc_assignment)
        # mec_plus_acc_makespan = get_makespan(time_costs, mec_plus_acc_assignment)
        # mec_plus_acc_energy_consumption = get_total_cost(energy_costs, mec_plus_acc_assignment)
        # mec_plus_acc_training_accuracy = get_total_cost(training_accuracies,
        #                                                 mec_plus_acc_assignment)
        mec_plus_acc_result = {"scheduler_name": scheduler_name,
                               "num_tasks": num_tasks,
                               "num_resources": num_resources,
                               "assignment": mec_plus_acc_assignment,
                               "num_selected_resources": mec_plus_acc_num_selected_resources,
                               "makespan": mec_plus_acc_makespan,
                               "energy_consumption": mec_plus_acc_energy_consumption,
                               "training_accuracy": mec_plus_acc_training_accuracy}
        scheduler_execution_result = mec_plus_acc_result
    elif scheduler_name == "ECMTC":
        # Run the ECMTC algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        time_limit = inf  # Time limit in seconds (deadline).
        ecmtc_assignment, ecmtc_energy_consumption, ecmtc_makespan \
            = ecmtc(num_resources,
                    num_tasks,
                    assignment_capacities,
                    time_costs,
                    energy_costs,
                    time_limit)
        ecmtc_num_selected_resources = get_num_selected_resources(ecmtc_assignment)
        # ecmtc_makespan = get_makespan(time_costs, ecmtc_assignment)
        # ecmtc_energy_consumption = get_total_cost(energy_costs, ecmtc_assignment)
        ecmtc_training_accuracy = get_total_cost(training_accuracies, ecmtc_assignment)
        ecmtc_result = {"scheduler_name": scheduler_name,
                        "num_tasks": num_tasks,
                        "num_resources": num_resources,
                        "assignment": ecmtc_assignment,
                        "num_selected_resources": ecmtc_num_selected_resources,
                        "makespan": ecmtc_makespan,
                        "energy_consumption": ecmtc_energy_consumption,
                        "training_accuracy": ecmtc_training_accuracy}
        scheduler_execution_result = ecmtc_result
    elif scheduler_name == "ECMTC+Acc":
        # Run the ECMTC+Acc algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        time_limit = inf  # Time limit in seconds (deadline).
        (ecmtc_plus_acc_assignment, ecmtc_plus_acc_energy_consumption, ecmtc_plus_acc_makespan,
         ecmtc_plus_acc_training_accuracy) \
            = ecmtc_plus_acc(num_resources,
                             num_tasks,
                             assignment_capacities,
                             time_costs,
                             energy_costs,
                             training_accuracies,
                             time_limit)
        ecmtc_plus_acc_num_selected_resources = get_num_selected_resources(ecmtc_plus_acc_assignment)
        # ecmtc_plus_acc_makespan = get_makespan(time_costs, ecmtc_plus_acc_assignment)
        # ecmtc_plus_acc_energy_consumption = get_total_cost(energy_costs, ecmtc_plus_acc_assignment)
        # ecmtc_plus_acc_training_accuracy = get_total_cost(training_accuracies,
        #                                                   ecmtc_plus_acc_assignment)
        ecmtc_plus_acc_result = {"scheduler_name": scheduler_name,
                                 "num_tasks": num_tasks,
                                 "num_resources": num_resources,
                                 "assignment": ecmtc_plus_acc_assignment,
                                 "num_selected_resources": ecmtc_plus_acc_num_selected_resources,
                                 "makespan": ecmtc_plus_acc_makespan,
                                 "energy_consumption": ecmtc_plus_acc_energy_consumption,
                                 "training_accuracy": ecmtc_plus_acc_training_accuracy}
        scheduler_execution_result = ecmtc_plus_acc_result
    return scheduler_execution_result


def run_for_fixed_tasks(schedulers_names: list,
                        execution_parameters: dict,
                        logger: Logger) -> None:
    """
    Runs experiments for a fixed number of tasks.
    Parameters
    ----------
    schedulers_names : list
        Schedulers names to run
    execution_parameters : dict
        Execution parameters
    logger: Logger
        Logging object
    """
    # Get the execution parameters.
    experiment_name = execution_parameters["experiment_name"]
    num_tasks = execution_parameters["num_tasks"]
    min_resources = execution_parameters["min_resources"]
    max_resources = execution_parameters["max_resources"]
    step_resources = execution_parameters["step_resources"]
    size_of_sample = execution_parameters["size_of_sample"]
    number_of_samples = execution_parameters["number_of_samples"]
    shuffle_seed = execution_parameters["shuffle_seed"]
    rng_resources_seed = execution_parameters["rng_resources_seed"]
    cost_function_verbose = execution_parameters["cost_function_verbose"]
    low_random_training_time = execution_parameters["low_random_training_time"]
    high_random_training_time = execution_parameters["high_random_training_time"]
    low_random_training_energy = execution_parameters["low_random_training_energy"]
    high_random_training_energy = execution_parameters["high_random_training_energy"]
    low_random_training_accuracy = execution_parameters["low_random_training_accuracy"]
    high_random_training_accuracy = execution_parameters["high_random_training_accuracy"]
    # Counting the rounds of the experiment to update the RNG seed.
    rounds = 0
    # Iterate over the number of resources.
    for num_resources in range(min_resources, max_resources+1, step_resources):
        # Number of resources message.
        print("\n{0}: Executing the '{1}' experiment for {2} resources..."
              .format(datetime.now(), experiment_name, num_resources))
        # Number of tasks to schedule message.
        print("\n{0}: Scheduling {1} tasks:".format(datetime.now(), num_tasks))
        # Set the lower and upper assignment limits arrays.
        lower_assignment_limits = zeros(shape=num_resources, dtype=int)
        upper_assignment_limits = full(shape=num_resources, fill_value=num_tasks, dtype=int)
        # Set the assignment capacities matrix.
        assignment_capacities = []
        for resource_index in range(num_resources):
            assignment_capacities_i = list(range(lower_assignment_limits[resource_index],
                                                 upper_assignment_limits[resource_index]))
            assignment_capacities.append(assignment_capacities_i)
        assignment_capacities = array(assignment_capacities)
        # Gather all samples for a given (num_tasks, scheduler) tuple.
        for sample in range(number_of_samples):
            # Set the shuffle seed and generate a random order of execution.
            seed(shuffle_seed + rounds)
            rounds += 1
            order = arange(len(schedulers_names))
            shuffle(order)
            # Gather samples for all schedulers.
            for order_index in order:
                # Initialize the cost matrices with zeros.
                time_costs = zeros(shape=(num_resources, num_tasks+1))
                energy_costs = zeros(shape=(num_resources, num_tasks+1))
                training_accuracies = zeros(shape=(num_resources, num_tasks+1))
                # Fill the cost matrices with costs based on a linear function.
                rng_resources_base_seed = rng_resources_seed
                for resource_index in range(num_resources):
                    # Fill the time_costs matrix.
                    create_linear_costs(rng_resources_base_seed,
                                        time_costs,
                                        resource_index,
                                        num_tasks,
                                        cost_function_verbose,
                                        low_random_training_time,
                                        high_random_training_time)
                    # Fill the energy_costs matrix.
                    create_linear_costs(rng_resources_base_seed,
                                        energy_costs,
                                        resource_index,
                                        num_tasks,
                                        cost_function_verbose,
                                        low_random_training_energy,
                                        high_random_training_energy)
                    # Fill the training_accuracies matrix.
                    create_linear_costs(rng_resources_base_seed,
                                        training_accuracies,
                                        resource_index,
                                        num_tasks,
                                        cost_function_verbose,
                                        low_random_training_accuracy,
                                        high_random_training_accuracy)
                    # Normalize the training_accuracies matrix (to avoid sum of training accuracies higher than 1.0).
                    training_accuracies = ((training_accuracies - training_accuracies.min()) /
                                           (training_accuracies - training_accuracies.min()).sum())
                    # Increment the base seed for resources.
                    rng_resources_base_seed += 1
                # Set the scheduler's execution parameters.
                scheduler_execution_parameters = {"index": order_index,
                                                  "scheduler_name": schedulers_names[order_index],
                                                  "num_tasks": num_tasks,
                                                  "num_resources": num_resources,
                                                  "time_costs": time_costs,
                                                  "energy_costs": energy_costs,
                                                  "training_accuracies": training_accuracies,
                                                  "assignment_capacities": assignment_capacities}
                # Gather one sample.
                timing = timeit(stmt=lambda: execute_scheduler(scheduler_execution_parameters),
                                number=size_of_sample)
                # Store the experiment result (timing information).
                experiment_result = ("{0},{1},{2},{3}"
                                     .format(schedulers_names[order_index], num_tasks, num_resources, timing))
                logger.store(experiment_result)


def run_experiment() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "linear_costs_fixed_tasks_timing"
    # Start message.
    print("{0}: Starting the '{1}' experiment...".format(datetime.now(), experiment_name))
    # Set the output CSV file to store the results.
    experiments_results_csv_file = Path("experiments_results/{0}_experiment_results.csv".format(experiment_name))
    # Create the parents directories of the output file (if not exist yet).
    experiments_results_csv_file.parent.mkdir(exist_ok=True, parents=True)
    # Remove the output file (if exists).
    experiments_results_csv_file.unlink(missing_ok=True)
    # Set the logger.
    logger_verbosity = False
    logger = Logger(experiments_results_csv_file, logger_verbosity)
    # Store the description of the experiments.
    experiments_description = __doc__
    logger.header(experiments_description)
    # Store the header of the output CSV file.
    experiments_csv_file_header = ("{0},{1},{2},{3}"
                                   .format("Scheduler_Name", "Num_Tasks", "Num_Resources", "Execution_Time"))
    logger.store(experiments_csv_file_header)
    # Set the execution parameters.
    scheduler_names = ["OLAR", "MC²MKP", "ELASTIC", "FedAECS",
                       "MEC", "MEC+Acc", "ECMTC", "ECMTC+Acc"]
    execution_parameters = {"experiment_name": experiment_name,
                            "num_tasks": 2000,
                            "min_resources": 10,
                            "max_resources": 100,
                            "step_resources": 15,
                            "size_of_sample": 5,
                            "number_of_samples": 20,
                            "shuffle_seed": 1000,
                            "rng_resources_seed": 100,
                            "cost_function_verbose": False,
                            "low_random_training_time": 1,
                            "high_random_training_time": 10,
                            "low_random_training_energy": 0.32,
                            "high_random_training_energy": 3.2,
                            "low_random_training_accuracy": 0.2,
                            "high_random_training_accuracy": 2.0}
    # Run the experiments.
    run_for_fixed_tasks(scheduler_names,
                        execution_parameters,
                        logger)
    # Finish logging.
    logger.finish()
    # Stop the performance counter.
    perf_counter_stop = perf_counter()
    # Get the elapsed time in seconds.
    elapsed_time_seconds = round((perf_counter_stop - perf_counter_start), 2)
    # End message.
    print("\n{0}: The '{1}' experiment was successfully executed! (Elapsed time: {2} {3})"
          .format(datetime.now(),
                  experiment_name,
                  elapsed_time_seconds,
                  "seconds" if elapsed_time_seconds != 1 else "second"))
    # Exit.
    exit(0)


if __name__ == '__main__':
    run_experiment()
