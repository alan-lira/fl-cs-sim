from datetime import datetime
from matplotlib.pyplot import figure, rcParams, savefig, xlabel, xticks, ylabel
from numpy import sum
from pandas import read_csv
from pathlib import Path
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
                          y=metric_name,
                          hue="Scheduler_Name",
                          style="Scheduler_Name",
                          dashes=False,
                          markers=True,
                          linewidth=2,
                          markersize=8)
            move_legend(ax,
                        "lower center",
                        bbox_to_anchor=(.5, 1),
                        ncol=3,
                        title=None,
                        frameon=True)
            output_figure_file = Path("fig-{0}-{1}-{2}.pdf".format(experiment_name, n_resources, metric_name.lower()))
            savefig(experiments_analysis_results_folder.joinpath(output_figure_file), bbox_inches="tight")


def compare_schedulers_performance(execution_parameters: dict) -> None:
    experiments_results_df = execution_parameters["experiments_results_df"]
    scheduler_names = execution_parameters["scheduler_names"]
    metrics_names = execution_parameters["metrics_names"]
    target_scheduler = execution_parameters["target_scheduler"]
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
                comparison_message = ("Number of times '{0}' provided a '{1}' value that is "
                                      "greater, equal, or smaller than '{2}': {3}, {4}, and {5}, respectively."
                                      .format(scheduler, metric_name, target_scheduler, greater, equal, less))
                print(comparison_message)


def run_experiment_analysis() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "nlogn_costs"
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
    num_resources = list(experiments_results_df["Num_Resources"].sort_values().unique())
    scheduler_names = list(experiments_results_df["Scheduler_Name"].sort_values().unique())
    metrics_names = experiments_results_df.columns.drop(["Scheduler_Name", "Num_Tasks", "Num_Resources"]).to_list()
    target_scheduler = "ECMTC"
    execution_parameters = {"experiment_name": experiment_name,
                            "experiments_analysis_results_folder": experiments_analysis_results_folder,
                            "experiments_results_df": experiments_results_df,
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
    # Check how many times other schedulers met the performance of the target scheduler.
    compare_schedulers_performance(execution_parameters)
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
