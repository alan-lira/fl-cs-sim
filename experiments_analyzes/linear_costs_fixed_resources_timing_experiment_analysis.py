from datetime import datetime
from matplotlib.pyplot import figure, rcParams, savefig, xlabel, xticks, ylabel, ylim, yticks, yscale
from numpy import mean, std, sum
from numpy.random import seed
from pandas import read_csv
from pathlib import Path
from scipy import stats
from seaborn import lineplot, move_legend, set_palette, set_theme
from shutil import rmtree
from time import perf_counter
from warnings import filterwarnings
filterwarnings("ignore")


def generate_experiments_results_figures(execution_parameters: dict) -> None:
    experiment_name = execution_parameters["experiment_name"]
    experiments_analysis_results_folder = execution_parameters["experiments_analysis_results_folder"]
    experiments_results_df = execution_parameters["experiments_results_df"]
    num_resources = execution_parameters["num_resources"]
    metrics_names = execution_parameters["metrics_names"]
    # Generate a figure for each (num_resources, metric_name) tuple.
    for n_resources in num_resources:
        for metric_name in metrics_names:
            if metric_name == "Num_Selected_Resources":
                y_label = metric_name.replace("_", " ").replace("Num", "Number of").capitalize()
            elif metric_name == "Execution_Time":
                # Each sample contains 5 measurements of the execution time.
                # Calculate their average execution time and convert it from seconds to microseconds (1 µs = 1e+6 s).
                size_of_sample = 5
                experiments_results_df["Execution_Time_Avg"] \
                    = experiments_results_df["Execution_Time"] * pow(10, 6) / size_of_sample
                y_label = metric_name.replace("_", " ").capitalize() + " (log scale)"
            else:
                y_label = metric_name.replace("_", " ").capitalize()
            figure(figsize=(6, 5))
            rcParams["axes.titlesize"] = 13
            rcParams["axes.labelsize"] = 13
            rcParams["xtick.labelsize"] = 13
            rcParams["ytick.labelsize"] = 13
            rcParams["legend.fontsize"] = 12
            xlabel("Number of tasks", fontsize=13)
            ylabel(y_label, fontsize=13)
            xticks(ticks=list(experiments_results_df["Num_Tasks"].sort_values().unique()), rotation=15)
            ax = lineplot(data=experiments_results_df[experiments_results_df.Num_Resources == n_resources],
                          x="Num_Tasks",
                          y="Execution_Time_Avg" if metric_name == "Execution_Time" else metric_name,
                          hue="Scheduler_Name",
                          style="Scheduler_Name",
                          dashes=False,
                          markers=True,
                          linewidth=2,
                          markersize=8)
            ylim(1, 100000000)
            yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 100000000])
            yscale("log")
            move_legend(ax,
                        "lower center",
                        bbox_to_anchor=(.5, 1),
                        ncol=3,
                        title=None,
                        frameon=True)
            output_figure_file = Path("fig-{0}-{1}-{2}.pdf".format(experiment_name, n_resources, metric_name.lower()))
            output_figure_file_full_path = experiments_analysis_results_folder.joinpath(output_figure_file)
            savefig(output_figure_file_full_path, bbox_inches="tight")
            print("Figure '{0}' was successfully generated.".format(output_figure_file_full_path))


def compare_schedulers_metrics_performance(execution_parameters: dict) -> None:
    experiments_results_df = execution_parameters["experiments_results_df"]
    scheduler_names = execution_parameters["scheduler_names"]
    metrics_names = execution_parameters["metrics_names"]
    target_scheduler = execution_parameters["target_scheduler"]
    print("\nSchedulers metrics performance comparison:\n")
    for metric_name in metrics_names:
        target_scheduler_cost = experiments_results_df[experiments_results_df["Scheduler_Name"] ==
                                                       target_scheduler][metric_name].reset_index(drop=True)
        for scheduler in scheduler_names:
            if scheduler != target_scheduler:
                other_cost = experiments_results_df[experiments_results_df["Scheduler_Name"] ==
                                                    scheduler][metric_name].reset_index(drop=True)
                greater = sum(other_cost > target_scheduler_cost)
                equal = sum(other_cost == target_scheduler_cost)
                less = sum(other_cost < target_scheduler_cost)
                comparison_message = ("- Number of times '{0}' provided a '{1}' value that is "
                                      "greater, equal, or smaller than '{2}': {3}, {4}, and {5}, respectively."
                                      .format(scheduler, metric_name, target_scheduler, greater, equal, less))
                print(comparison_message)


def show_execution_time_distribution(execution_parameters: dict) -> None:
    experiments_results_df = execution_parameters["experiments_results_df"]
    num_tasks = execution_parameters["num_tasks"]
    scheduler_names = execution_parameters["scheduler_names"]
    print("\nExecution time distribution for scheduling {0} and {1} tasks with different schedulers:\n"
          .format(min(num_tasks), max(num_tasks)))
    # Each sample contains 5 measurements of the execution time.
    # Calculate their average execution time and convert it from seconds to microseconds (1 µs = 1e+6 s).
    size_of_sample = 5
    experiments_results_df["Execution_Time_Avg"] \
        = experiments_results_df["Execution_Time"] * pow(10, 6) / size_of_sample
    for scheduler in scheduler_names:
        for tasks in (min(num_tasks), max(num_tasks)):
            resulting_df = experiments_results_df[(experiments_results_df["Scheduler_Name"] == scheduler) &
                                                  (experiments_results_df["Num_Tasks"] == tasks)].Execution_Time_Avg
            print("- Scheduler '{0}' with {1} tasks:\n{2}\n"
                  .format(scheduler,
                          tasks,
                          resulting_df.describe()))


def perform_kolmogorov_smirnov_tests(execution_parameters: dict) -> None:
    experiments_results_df = execution_parameters["experiments_results_df"]
    num_tasks = execution_parameters["num_tasks"]
    scheduler_names = execution_parameters["scheduler_names"]
    # Get the step of tasks.
    step_tasks = num_tasks[1] - num_tasks[0]
    print("\nKolmogorov-Smirnov tests for scheduling from {0} to {1} tasks with different schedulers:"
          .format(min(num_tasks), max(num_tasks)))
    print("Note: Results with p-values < 0.05 means that they do not follow normal distributions.\n")
    # Each sample contains 5 measurements of the execution time.
    # Calculate their average execution time and convert it from seconds to microseconds (1 µs = 1e+6 s).
    size_of_sample = 5
    experiments_results_df["Execution_Time_Avg"] \
        = experiments_results_df["Execution_Time"] * pow(10, 6) / size_of_sample
    # Set the seed.
    seed(2022)
    for scheduler in scheduler_names:
        for tasks in range(min(num_tasks), max(num_tasks)+1, step_tasks):
            resulting_df \
                = list(experiments_results_df[(experiments_results_df["Scheduler_Name"] == scheduler) &
                                              (experiments_results_df["Num_Tasks"] == tasks)].Execution_Time_Avg)
            print("- Scheduler '{0}' with {1} tasks:\n{2}\n"
                  .format(scheduler,
                          tasks,
                          stats.kstest(resulting_df, "norm", args=(mean(resulting_df), std(resulting_df)))))


def run_experiment_analysis() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "linear_costs_fixed_resources_timing"
    # Start message.
    print("{0}: Starting the '{1}' experiment's analysis...".format(datetime.now(), experiment_name))
    # Get the experiments results CSV file.
    experiments_results_csv_file = Path("experiments_results/{0}_experiment_results.csv".format(experiment_name))
    # Set the experiments analyzes results folder.
    experiments_analysis_results_folder = Path("experiments_analyzes_results/{0}".format(experiment_name))
    # Remove the output folder and its contents (if exists).
    if experiments_analysis_results_folder.is_dir():
        rmtree(experiments_analysis_results_folder)
    # Create the parents directories of the output file (if not exist yet).
    experiments_analysis_results_folder.parent.mkdir(exist_ok=True, parents=True)
    # Create the output folder.
    experiments_analysis_results_folder.mkdir(exist_ok=True, parents=True)
    # Load the dataframe from the experiments results CSV file.
    experiments_results_df = read_csv(experiments_results_csv_file, comment="#")
    # Set the execution parameters.
    num_tasks = list(experiments_results_df["Num_Tasks"].sort_values().unique())
    num_resources = list(experiments_results_df["Num_Resources"].sort_values().unique())
    scheduler_names = list(experiments_results_df["Scheduler_Name"].sort_values().unique())
    metrics_names = experiments_results_df.columns.drop(["Scheduler_Name", "Num_Tasks", "Num_Resources"]).to_list()
    target_scheduler = "ECMTC"
    execution_parameters = {"experiment_name": experiment_name,
                            "experiments_analysis_results_folder": experiments_analysis_results_folder,
                            "experiments_results_df": experiments_results_df,
                            "num_tasks": num_tasks,
                            "num_resources": num_resources,
                            "scheduler_names": scheduler_names,
                            "metrics_names": metrics_names,
                            "target_scheduler": target_scheduler}
    # Set the visual theme for all matplotlib and seaborn plots.
    set_theme(style="whitegrid")
    # Set the matplotlib color cycle using a seaborn palette.
    set_palette("hls", len(scheduler_names))
    # Generate the experiments results figures.
    generate_experiments_results_figures(execution_parameters)
    # Check how many times other schedulers met the performance of the target scheduler for a set of metrics.
    compare_schedulers_metrics_performance(execution_parameters)
    # Show the execution time distribution per scheduler per number of tasks.
    show_execution_time_distribution(execution_parameters)
    # Perform Kolmogorov-Smirnov tests.
    perform_kolmogorov_smirnov_tests(execution_parameters)
    # Stop the performance counter.
    perf_counter_stop = perf_counter()
    # Get the elapsed time in seconds.
    elapsed_time_seconds = round((perf_counter_stop - perf_counter_start), 2)
    # End message.
    print("\n{0}: The '{1}' experiment's analysis was successfully executed! (Elapsed time: {2} {3})"
          .format(datetime.now(),
                  experiment_name,
                  elapsed_time_seconds,
                  "seconds" if elapsed_time_seconds != 1 else "second"))
    # Exit.
    exit(0)


if __name__ == '__main__':
    run_experiment_analysis()
