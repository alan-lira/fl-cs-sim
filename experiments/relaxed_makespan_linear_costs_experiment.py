"""
# Description of the Experiment:
#  - We generate the costs to up to 5000 tasks for 10 and 100 resources.
#  - All resources costs follow linear functions with RNG seeds starting at 100.
#  - Time costs were generated using uniformly distributed values ranging from 1 to 10.
#  - Energy costs were generated using uniformly distributed values ranging from 0.32 to 3.2.
#  - Accuracy costs were generated using uniformly distributed values ranging from 0.2 to 2.0 (then normalized).
#  - We schedule from 1000 to 5000 tasks in increments of 100.
#  - We run MEC followed by ECMTC.
#  - We use no lower or upper task assignment limits for resources.
#  - We use relaxation percentages of 0.25 (25%); 0.50 (50%); 0.75 (75%); 1.0 (100%);
#    1.25 (125%); 1.50 (150%); 1.75 (175%); and 2.0 (200%) for the minimal makespan found
#    to evaluate the potential energy consumption reductions.
#  - Every result is verified and logged to a CSV file.
"""

from datetime import datetime
from math import inf
from multiprocessing import cpu_count, Manager, Pool, Queue
from numpy import array, full, sum, zeros
from pathlib import Path
from time import perf_counter

from devices.linear_cost_device import create_linear_costs
from schedulers.ecmtc import ecmtc
from schedulers.mec import mec
from util.experiment_util import get_num_selected_resources, get_total_cost, check_total_assigned
from util.logger_util import Logger


def execute_scheduler(scheduler_execution_parameters: dict) -> dict:
    # Get the scheduler's execution parameters.
    index = scheduler_execution_parameters["index"]
    scheduler_name = scheduler_execution_parameters["scheduler_name"]
    latter_scheduler_name = scheduler_execution_parameters["latter_scheduler_name"]
    num_tasks = scheduler_execution_parameters["num_tasks"]
    num_resources = scheduler_execution_parameters["num_resources"]
    time_costs = scheduler_execution_parameters["time_costs"]
    energy_costs = scheduler_execution_parameters["energy_costs"]
    training_accuracies = scheduler_execution_parameters["training_accuracies"]
    assignment_capacities = scheduler_execution_parameters["assignment_capacities"]
    scheduler_execution_result = None
    if scheduler_name == "MEC":
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        # Run the MEC algorithm.
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
                      "latter_scheduler_name": latter_scheduler_name,
                      "num_tasks": num_tasks,
                      "num_resources": num_resources,
                      "assignment": mec_assignment,
                      "num_selected_resources": mec_num_selected_resources,
                      "time_limit": inf,
                      "makespan_relaxation_percentage": 0,
                      "makespan": mec_makespan,
                      "energy_consumption": mec_energy_consumption,
                      "training_accuracy": mec_training_accuracy}
        scheduler_execution_result = mec_result
    elif scheduler_name == "ECMTC":
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        # Get the base makespan.
        base_makespan = scheduler_execution_parameters["base_makespan"]
        # Get the makespan relaxation percentage.
        makespan_relaxation_percentage = scheduler_execution_parameters["makespan_relaxation_percentage"]
        # Run the ECMTC algorithm.
        time_limit = base_makespan * (1 + makespan_relaxation_percentage)
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
                        "latter_scheduler_name": latter_scheduler_name,
                        "num_tasks": num_tasks,
                        "num_resources": num_resources,
                        "assignment": ecmtc_assignment,
                        "num_selected_resources": ecmtc_num_selected_resources,
                        "time_limit": time_limit,
                        "makespan_relaxation_percentage": makespan_relaxation_percentage,
                        "makespan": ecmtc_makespan,
                        "energy_consumption": ecmtc_energy_consumption,
                        "training_accuracy": ecmtc_training_accuracy}
        scheduler_execution_result = ecmtc_result
    return scheduler_execution_result


def check_and_store(scheduler_result: dict,
                    logger: Logger) -> None:
    """
    Checks if the results are correct and stores them in the logger.

    Parameters
    ----------
    scheduler_result : dict
        Scheduler assignment result
    logger: Logger
        Logging object
    """
    # Get the scheduler assignment result values.
    scheduler_name = scheduler_result["scheduler_name"]
    num_tasks = scheduler_result["num_tasks"]
    num_resources = scheduler_result["num_resources"]
    assignment = scheduler_result["assignment"]
    num_selected_resources = scheduler_result["num_selected_resources"]
    time_limit = scheduler_result["time_limit"]
    makespan_relaxation_percentage = scheduler_result["makespan_relaxation_percentage"]
    makespan = scheduler_result["makespan"]
    energy_consumption = scheduler_result["energy_consumption"]
    training_accuracy = scheduler_result["training_accuracy"]
    # Check if all tasks were assigned.
    if not check_total_assigned(num_tasks, assignment):
        failed_assignment_message = ("Attention: {0} failed to assign {1} tasks to {2} resources ({3} were assigned)!"
                                     .format(scheduler_name, num_tasks, num_resources, sum(assignment)))
        print(failed_assignment_message)
    # Store the experiment result.
    experiment_result = ("{0},{1},{2},{3},{4},{5},{6}"
                         .format(scheduler_name, num_tasks, num_resources, time_limit,
                                 makespan_relaxation_percentage, makespan, energy_consumption))
    logger.store(experiment_result)


def queue_consumer(queue: Queue) -> list:
    queue_items = []
    while True:
        queue_item = queue.get()
        if queue_item == "finish_consumer":
            return queue_items
        queue_items.append(queue_item)


def queue_producer(scheduler_execution_parameters: dict,
                   queue: Queue) -> None:
    # Execute the scheduler.
    producer_result = execute_scheduler(scheduler_execution_parameters)
    # Put the scheduler's execution result to the queue.
    queue.put(producer_result)


def send_finish_consumer_message_to_queue(queue: Queue) -> None:
    queue.put("finish_consumer")


def close_pool(pool: Pool) -> None:
    pool.close()
    pool.join()


def run_for_n_resources(num_resources: int,
                        schedulers_names: list,
                        execution_parameters: dict,
                        logger: Logger) -> None:
    """
    Runs experiments for a number of resources.
    Parameters
    ----------
    num_resources : int
        Number of resources
    schedulers_names : list
        Schedulers names to run
    execution_parameters : dict
        Execution parameters
    logger: Logger
        Logging object
    """
    # Get the execution parameters.
    experiment_name = execution_parameters["experiment_name"]
    use_multiprocessing = execution_parameters["use_multiprocessing"]
    num_queue_consumers = execution_parameters["num_queue_consumers"]
    num_queue_producers = execution_parameters["num_queue_producers"]
    min_tasks = execution_parameters["min_tasks"]
    max_tasks = execution_parameters["max_tasks"]
    step_tasks = execution_parameters["step_tasks"]
    makespan_relaxation_percentages = execution_parameters["makespan_relaxation_percentages"]
    rng_resources_seed = execution_parameters["rng_resources_seed"]
    cost_function_verbose = execution_parameters["cost_function_verbose"]
    low_random_training_time = execution_parameters["low_random_training_time"]
    high_random_training_time = execution_parameters["high_random_training_time"]
    low_random_training_energy = execution_parameters["low_random_training_energy"]
    high_random_training_energy = execution_parameters["high_random_training_energy"]
    low_random_training_accuracy = execution_parameters["low_random_training_accuracy"]
    high_random_training_accuracy = execution_parameters["high_random_training_accuracy"]
    # Multiprocessing status message.
    print("\n{0}: Multiprocessing is {1}!".format(datetime.now(),
                                                  "enabled" if use_multiprocessing else "disabled"))
    # Number of resources message.
    print("\n{0}: Executing the '{1}' experiment for {2} resources..."
          .format(datetime.now(), experiment_name, num_resources))
    # Initialize the cost matrices with zeros.
    time_costs = zeros(shape=(num_resources, max_tasks+1))
    energy_costs = zeros(shape=(num_resources, max_tasks+1))
    training_accuracies = zeros(shape=(num_resources, max_tasks+1))
    # Fill the cost matrices with costs based on a linear function.
    rng_resources_base_seed = rng_resources_seed
    for resource_index in range(num_resources):
        # Fill the time_costs matrix.
        create_linear_costs(rng_resources_base_seed,
                            time_costs,
                            resource_index,
                            max_tasks,
                            cost_function_verbose,
                            low_random_training_time,
                            high_random_training_time)
        # Fill the energy_costs matrix.
        create_linear_costs(rng_resources_base_seed,
                            energy_costs,
                            resource_index,
                            max_tasks,
                            cost_function_verbose,
                            low_random_training_energy,
                            high_random_training_energy)
        # Fill the training_accuracies matrix.
        create_linear_costs(rng_resources_base_seed,
                            training_accuracies,
                            resource_index,
                            max_tasks,
                            cost_function_verbose,
                            low_random_training_accuracy,
                            high_random_training_accuracy)
        # Normalize the training_accuracies matrix (to avoid sum of training accuracies higher than 1.0).
        training_accuracies = ((training_accuracies - training_accuracies.min()) /
                               (training_accuracies.max() - training_accuracies.min()))
        # Increment the base seed for resources.
        rng_resources_base_seed += 1
    # Iterate over the number of tasks to assign.
    for num_tasks in range(min_tasks, max_tasks+1, step_tasks):
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
        # Whether to use multiprocessing or not.
        if use_multiprocessing:
            # Load the multiprocessing elements (queue, pool).
            # Set the queue.
            queue = Manager().Queue()
            # Set the pool.
            pool = Pool(num_queue_consumers + num_queue_producers)
            # Start the queue consumers.
            all_queue_items = []
            for _ in range(num_queue_consumers):
                queue_items = pool.apply_async(queue_consumer, (queue,))
                all_queue_items.append(queue_items)
            async_results = []
            # Iterate over the schedulers that will assign the tasks.
            for index, scheduler_name in enumerate(schedulers_names):
                # Get the first scheduler's name, which will define the base makespan.
                first_scheduler_name = str(scheduler_name).split("_to_")[0]
                # Get the latter scheduler's name, which will use the first scheduler's makespan during relaxations.
                latter_scheduler_name = str(scheduler_name).split("_to_")[1]
                # Set the first scheduler's execution parameters.
                first_scheduler_execution_parameters \
                    = {"index": index,
                       "scheduler_name": first_scheduler_name,
                       "latter_scheduler_name": latter_scheduler_name,
                       "num_tasks": num_tasks,
                       "num_resources": num_resources,
                       "time_costs": time_costs,
                       "energy_costs": energy_costs,
                       "training_accuracies": training_accuracies,
                       "assignment_capacities": assignment_capacities}
                # Launch asynchronous tasks for the queue producers.
                async_result = pool.apply_async(queue_producer, (first_scheduler_execution_parameters, queue))
                async_results.append(async_result)
            # Collect results from the producers through the pool result queue.
            for async_result in async_results:
                async_result.get()
            # Send the 'finish_consumer' messages to the queue.
            for _ in range(num_queue_consumers):
                send_finish_consumer_message_to_queue(queue)
            # Get the producers' results stored by the consumers.
            for queue_item in all_queue_items:
                producer_results = queue_item.get()
                for producer_result in producer_results:
                    # Check and store the first scheduler's execution result.
                    check_and_store(producer_result, logger)
                    # Get the latter scheduler's name, which will use the first scheduler's makespan during relaxations.
                    latter_scheduler_name = producer_result["latter_scheduler_name"]
                    # Get the first scheduler's makespan and set as the base makespan.
                    base_makespan = producer_result["makespan"]
                    # Iterate over the makespan relaxation percentages.
                    for makespan_relaxation_percentage in makespan_relaxation_percentages:
                        # Number of tasks to schedule plus makespan relaxation percentage message.
                        print("\n{0}: Scheduling {1} tasks with makespan relaxation of {2}%:"
                              .format(datetime.now(), num_tasks, makespan_relaxation_percentage * 100))
                        # Set the latter scheduler's execution parameters.
                        latter_scheduler_execution_parameters \
                            = {"index": None,
                               "scheduler_name": latter_scheduler_name,
                               "latter_scheduler_name": None,
                               "num_tasks": num_tasks,
                               "num_resources": num_resources,
                               "time_costs": time_costs,
                               "energy_costs": energy_costs,
                               "training_accuracies": training_accuracies,
                               "assignment_capacities": assignment_capacities,
                               "base_makespan": base_makespan,
                               "makespan_relaxation_percentage": makespan_relaxation_percentage}
                        # Execute the latter scheduler.
                        latter_scheduler_execution_result = execute_scheduler(latter_scheduler_execution_parameters)
                        # Check and store the latter scheduler's execution result.
                        check_and_store(latter_scheduler_execution_result, logger)
            # Close the pool.
            close_pool(pool)
        else:
            # Iterate over the schedulers that will assign the tasks.
            for index, scheduler_name in enumerate(schedulers_names):
                # Get the first scheduler's name, which will define the base makespan.
                first_scheduler_name = str(scheduler_name).split("_to_")[0]
                # Get the latter scheduler's name, which will use the first scheduler's makespan during relaxations.
                latter_scheduler_name = str(scheduler_name).split("_to_")[1]
                # Set the first scheduler's execution parameters.
                first_scheduler_execution_parameters \
                    = {"index": index,
                       "scheduler_name": first_scheduler_name,
                       "latter_scheduler_name": latter_scheduler_name,
                       "num_tasks": num_tasks,
                       "num_resources": num_resources,
                       "time_costs": time_costs,
                       "energy_costs": energy_costs,
                       "training_accuracies": training_accuracies,
                       "assignment_capacities": assignment_capacities}
                # Execute the first scheduler.
                first_scheduler_execution_result = execute_scheduler(first_scheduler_execution_parameters)
                # Check and store the first scheduler's execution result.
                check_and_store(first_scheduler_execution_result, logger)
                # Get the first scheduler's makespan and set as the base makespan.
                base_makespan = first_scheduler_execution_result["makespan"]
                # Iterate over the makespan relaxation percentages.
                for makespan_relaxation_percentage in makespan_relaxation_percentages:
                    # Number of tasks to schedule plus makespan relaxation percentage message.
                    print("\n{0}: Scheduling {1} tasks with makespan relaxation of {2}%:"
                          .format(datetime.now(), num_tasks, makespan_relaxation_percentage * 100))
                    # Set the latter scheduler's execution parameters.
                    latter_scheduler_execution_parameters \
                        = {"index": index,
                           "scheduler_name": latter_scheduler_name,
                           "latter_scheduler_name": None,
                           "num_tasks": num_tasks,
                           "num_resources": num_resources,
                           "time_costs": time_costs,
                           "energy_costs": energy_costs,
                           "training_accuracies": training_accuracies,
                           "assignment_capacities": assignment_capacities,
                           "base_makespan": base_makespan,
                           "makespan_relaxation_percentage": makespan_relaxation_percentage}
                    # Execute the latter scheduler.
                    latter_scheduler_execution_result = execute_scheduler(latter_scheduler_execution_parameters)
                    # Check and store the latter scheduler's execution result.
                    check_and_store(latter_scheduler_execution_result, logger)


def run_experiment() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "relaxed_makespan_linear_costs"
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
    experiments_csv_file_header = ("{0},{1},{2},{3},{4},{5},{6}"
                                   .format("Scheduler_Name", "Num_Tasks", "Num_Resources", "Time_Limit",
                                           "Makespan_Relaxation_Percentage", "Makespan", "Energy_Consumption"))
    logger.store(experiments_csv_file_header)
    # Set the execution parameters.
    num_resources = [10, 100]
    scheduler_names = ["MEC_to_ECMTC"]
    num_queue_consumers = 1
    num_queue_producers = min(len(scheduler_names), cpu_count() - num_queue_consumers)
    makespan_relaxation_percentages = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]
    execution_parameters = {"experiment_name": experiment_name,
                            "use_multiprocessing": False,
                            "num_queue_consumers": num_queue_consumers,
                            "num_queue_producers": num_queue_producers,
                            "min_tasks": 1000,
                            "max_tasks": 5000,
                            "step_tasks": 100,
                            "makespan_relaxation_percentages": makespan_relaxation_percentages,
                            "rng_resources_seed": 100,
                            "cost_function_verbose": False,
                            "low_random_training_time": 1,
                            "high_random_training_time": 10,
                            "low_random_training_energy": 0.32,
                            "high_random_training_energy": 3.2,
                            "low_random_training_accuracy": 0.2,
                            "high_random_training_accuracy": 2.0}
    # Run the experiments.
    for n_resources in num_resources:
        run_for_n_resources(n_resources,
                            scheduler_names,
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
