"""
# Description of the Experiment:
#  - We generate the costs to up to 5000 tasks for num_resources
#    where num_resources ∈ {10; 25; 40; 55; 70; 85; 100}.
#  - All costs follow linear functions with RNG seeds [100..199].
#  - We schedule from 1000 to 5000 tasks in increments of 100.
#  - We run OLAR_Adapted; MC²MKP_Adapted; ELASTIC_Adapted; FedAECS_Adapted;
#    MEC; MEC_With_Accuracy; ECMTC; and ECMTC_With_Accuracy schedulers.
#  - We use no lower or upper limits.
#  - Every result is verified and logged to a CSV file.
"""

from datetime import datetime
from multiprocessing import Manager, Pool, Queue
from numpy import array, expand_dims, full, inf, sum, zeros
from pathlib import Path
from time import perf_counter

from devices.linear_cost_device import create_linear_costs
from schedulers.ecmtc import ecmtc
from schedulers.ecmtc_with_accuracy import ecmtc_with_accuracy
from schedulers.elastic_adapted import elastic_adapted_client_selection_algorithm
from schedulers.fedaecs_adapted import fedaecs_adapted
from schedulers.mc2mkp_adapted import mc2mkp_adapted
from schedulers.mec import mec
from schedulers.mec_with_accuracy import mec_with_accuracy
from schedulers.olar_adapted import olar_adapted
from util.experiment_util import get_num_selected_resources, get_makespan, get_total_cost, check_total_assigned
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
    if scheduler_name == "OLAR_Adapted":
        # Run the OLAR_Adapted algorithm.
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
    elif scheduler_name == "MC²MKP_Adapted":
        # Run the MC²MKP_Adapted algorithm.
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
    elif scheduler_name == "ELASTIC_Adapted":
        # Run the ELASTIC_Adapted algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        τ = inf  # Time limit in seconds (deadline).
        α = 1  # α (0 ≤ α ≤ 1) is a parameter to adjust the weights of the two objectives:
        # minimizing the energy consumption of the selected clients and
        # maximizing the number of selected clients for each BS.
        # α == 0 ----------> ni = -1, ∀i ∈ I.
        # α > 0 && α < 1 --> ni = α * (E_comp_i + E_up_i + 1) - 1, ∀i ∈ I.
        # α == 1 ----------> ni = E_comp_i + E_up_i, ∀i ∈ I.
        assignment_capacities_elastic = []
        # Divide the tasks as equally as possible.
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
    elif scheduler_name == "FedAECS_Adapted":
        # Run the FedAECS_Adapted algorithm.
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
    elif scheduler_name == "MEC_With_Accuracy":
        # Run the MEC_With_Accuracy algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        (mec_with_accuracy_assignment, mec_with_accuracy_makespan, mec_with_accuracy_energy_consumption,
         mec_with_accuracy_training_accuracy) \
            = mec_with_accuracy(num_resources,
                                num_tasks,
                                assignment_capacities,
                                time_costs,
                                energy_costs,
                                training_accuracies)
        mec_with_accuracy_num_selected_resources = get_num_selected_resources(mec_with_accuracy_assignment)
        # mec_with_accuracy_makespan = get_makespan(time_costs, mec_with_accuracy_assignment)
        # mec_with_accuracy_energy_consumption = get_total_cost(energy_costs, mec_with_accuracy_assignment)
        # mec_with_accuracy_training_accuracy = get_total_cost(training_accuracies,
        #                                                      mec_with_accuracy_assignment)
        mec_with_accuracy_result = {"scheduler_name": scheduler_name,
                                    "num_tasks": num_tasks,
                                    "num_resources": num_resources,
                                    "assignment": mec_with_accuracy_assignment,
                                    "num_selected_resources": mec_with_accuracy_num_selected_resources,
                                    "makespan": mec_with_accuracy_makespan,
                                    "energy_consumption": mec_with_accuracy_energy_consumption,
                                    "training_accuracy": mec_with_accuracy_training_accuracy}
        scheduler_execution_result = mec_with_accuracy_result
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
    elif scheduler_name == "ECMTC_With_Accuracy":
        # Run the ECMTC_With_Accuracy algorithm.
        print("{0}: {1}. Using {2}...".format(datetime.now(), index, scheduler_name))
        time_limit = inf  # Time limit in seconds (deadline).
        (ecmtc_with_accuracy_assignment, ecmtc_with_accuracy_energy_consumption, ecmtc_with_accuracy_makespan,
         ecmtc_with_accuracy_training_accuracy) \
            = ecmtc_with_accuracy(num_resources,
                                  num_tasks,
                                  assignment_capacities,
                                  time_costs,
                                  energy_costs,
                                  training_accuracies,
                                  time_limit)
        ecmtc_with_accuracy_num_selected_resources = get_num_selected_resources(ecmtc_with_accuracy_assignment)
        # ecmtc_with_accuracy_makespan = get_makespan(time_costs, ecmtc_with_accuracy_assignment)
        # ecmtc_with_accuracy_energy_consumption = get_total_cost(energy_costs, ecmtc_with_accuracy_assignment)
        # ecmtc_with_accuracy_training_accuracy = get_total_cost(training_accuracies,
        #                                                        ecmtc_with_accuracy_assignment)
        ecmtc_with_accuracy_result = {"scheduler_name": scheduler_name,
                                      "num_tasks": num_tasks,
                                      "num_resources": num_resources,
                                      "assignment": ecmtc_with_accuracy_assignment,
                                      "num_selected_resources": ecmtc_with_accuracy_num_selected_resources,
                                      "makespan": ecmtc_with_accuracy_makespan,
                                      "energy_consumption": ecmtc_with_accuracy_energy_consumption,
                                      "training_accuracy": ecmtc_with_accuracy_training_accuracy}
        scheduler_execution_result = ecmtc_with_accuracy_result
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
                         .format(scheduler_name, num_tasks, num_resources, num_selected_resources,
                                 makespan, energy_consumption, training_accuracy))
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
    rng_seed_resources = execution_parameters["rng_seed_resources"]
    cost_function_verbose = execution_parameters["cost_function_verbose"]
    low_random = execution_parameters["low_random"]
    high_random = execution_parameters["high_random"]
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
    base_seed = rng_seed_resources
    for resource_index in range(num_resources):
        # Fill the time_costs matrix.
        create_linear_costs(base_seed, time_costs, resource_index, max_tasks,
                            cost_function_verbose, low_random, high_random)
        # Fill the energy_costs matrix.
        create_linear_costs(base_seed, energy_costs, resource_index, max_tasks,
                            cost_function_verbose, (0.32 * low_random), (0.32 * high_random))
        # Fill the training_accuracies matrix.
        create_linear_costs(base_seed, training_accuracies, resource_index, max_tasks,
                            cost_function_verbose, (0.00002 * low_random), (0.00002 * high_random))
        base_seed += 1
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
                # Set the scheduler's execution parameters.
                scheduler_execution_parameters = {"index": index,
                                                  "scheduler_name": scheduler_name,
                                                  "num_tasks": num_tasks,
                                                  "num_resources": num_resources,
                                                  "time_costs": time_costs,
                                                  "energy_costs": energy_costs,
                                                  "training_accuracies": training_accuracies,
                                                  "assignment_capacities": assignment_capacities}
                # Launch asynchronous tasks for the queue producers.
                async_result = pool.apply_async(queue_producer, (scheduler_execution_parameters, queue))
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
                    # Check and store the scheduler's execution result.
                    check_and_store(producer_result, logger)
            # Close the pool.
            close_pool(pool)
        else:
            # Iterate over the schedulers that will assign the tasks.
            for index, scheduler_name in enumerate(schedulers_names):
                # Set the scheduler's execution parameters.
                scheduler_execution_parameters = {"index": index,
                                                  "scheduler_name": scheduler_name,
                                                  "num_tasks": num_tasks,
                                                  "num_resources": num_resources,
                                                  "time_costs": time_costs,
                                                  "energy_costs": energy_costs,
                                                  "training_accuracies": training_accuracies,
                                                  "assignment_capacities": assignment_capacities}
                # Execute the scheduler.
                scheduler_execution_result = execute_scheduler(scheduler_execution_parameters)
                # Check and store the scheduler's execution result.
                check_and_store(scheduler_execution_result, logger)


def run_experiment() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "linear_costs"
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
                                   .format("Scheduler_Name", "Num_Tasks", "Num_Resources",
                                           "Num_Selected_Resources", "Makespan", "Energy_Consumption",
                                           "Training_Accuracy"))
    logger.store(experiments_csv_file_header)
    # Set the execution parameters.
    num_resources_list = [10, 25, 40, 55, 70, 85, 100]
    scheduler_names = ["OLAR_Adapted", "MC²MKP_Adapted", "ELASTIC_Adapted", "FedAECS_Adapted",
                       "MEC", "MEC_With_Accuracy", "ECMTC", "ECMTC_With_Accuracy"]
    execution_parameters = {"experiment_name": experiment_name,
                            "use_multiprocessing": True,
                            "num_queue_consumers": 1,
                            "num_queue_producers": 8,
                            "min_tasks": 1000,
                            "max_tasks": 5000,
                            "step_tasks": 100,
                            "rng_seed_resources": 100,
                            "cost_function_verbose": False,
                            "low_random": 1,
                            "high_random": 10}
    # Run the experiments.
    for num_resources in num_resources_list:
        run_for_n_resources(num_resources,
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
