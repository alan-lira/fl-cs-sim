from datetime import datetime
from matplotlib.pyplot import figure, rcParams, savefig, xlabel, xticks, ylabel
from numpy import sum
from pandas import read_csv
from pathlib import Path
from seaborn import color_palette, lineplot, move_legend, set_palette, set_theme
from shutil import rmtree
from time import perf_counter
from warnings import filterwarnings
filterwarnings("ignore")


def generate_experiments_results_figures(execution_parameters: dict) -> None:
    experiment_name = execution_parameters["experiment_name"]
    experiments_analysis_results_folder = execution_parameters["experiments_analysis_results_folder"]
    experiments_results_df = execution_parameters["experiments_results_df"]
    num_resources = execution_parameters["num_resources"]
    scheduler_names = execution_parameters["scheduler_names"]
    metrics_names = execution_parameters["metrics_names"]
    makespan_relaxation_percentages = execution_parameters["makespan_relaxation_percentages"]
    target_scheduler = execution_parameters["target_scheduler"]
    x_ticks = execution_parameters["x_ticks"]
    alpha = execution_parameters["alpha"]
    theme_style = execution_parameters["theme_style"]
    line_colors = execution_parameters["line_colors"]
    line_sizes = execution_parameters["line_sizes"]
    # Set the visual theme for all matplotlib and seaborn plots.
    set_theme(style=theme_style)
    # Set the matplotlib color cycle using a seaborn palette.
    colors = color_palette(line_colors)
    set_palette(colors)
    # Generate a figure for each (metric_name, num_resources, makespan_relaxation_percentage) tuple.
    for metric_name in metrics_names:
        for n_resources in num_resources:
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
            xticks(ticks=x_ticks, rotation=15)
            ylabel(y_label, fontsize=13)
            base_scheduler_data \
                = experiments_results_df[(experiments_results_df.Num_Resources == n_resources) &
                                         (experiments_results_df.Scheduler_Name == target_scheduler)]
            ax = lineplot(data=base_scheduler_data,
                          x="Num_Tasks",
                          y=metric_name,
                          hue_order=scheduler_names,
                          style="Scheduler_Name",
                          dashes=False,
                          markers=True,
                          alpha=alpha,
                          size="Scheduler_Name",
                          sizes=line_sizes,
                          markersize=4)
            move_legend(ax,
                        "lower center",
                        bbox_to_anchor=(.5, 1),
                        ncol=3,
                        title=None,
                        frameon=True)
            for makespan_relaxation_percentage in makespan_relaxation_percentages:
                relaxed_makespan_scheduler_data \
                    = experiments_results_df[(experiments_results_df.Num_Resources == n_resources) &
                                             (experiments_results_df.Makespan_Relaxation_Percentage == makespan_relaxation_percentage) &
                                             experiments_results_df.Makespan_Relaxation_Percentage > 0 &
                                             (experiments_results_df.Scheduler_Name != target_scheduler)]
                scheduler_name_with_relaxed_makespan \
                    = (relaxed_makespan_scheduler_data.Scheduler_Name +
                       " ({0}% relax.)".format(makespan_relaxation_percentage * 100))
                relaxed_makespan_scheduler_data.Scheduler_Name = scheduler_name_with_relaxed_makespan
                ax = lineplot(data=relaxed_makespan_scheduler_data,
                              x="Num_Tasks",
                              y=metric_name,
                              hue_order=scheduler_names,
                              style="Scheduler_Name",
                              dashes=False,
                              markers=True,
                              alpha=alpha,
                              size="Scheduler_Name",
                              sizes=line_sizes,
                              markersize=4)
                move_legend(ax,
                            "lower center",
                            bbox_to_anchor=(.5, 1),
                            ncol=3,
                            title=None,
                            frameon=True)
            output_figure_file = Path("fig_{0}_{1}_resources_{2}.pdf"
                                      .format(experiment_name,
                                              n_resources,
                                              metric_name.lower()))
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


def run_experiment_analysis() -> None:
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Set the experiment name.
    experiment_name = "relaxed_makespan_random_costs"
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
    metrics_names = ["Makespan", "Energy_Consumption"]
    makespan_relaxation_percentages \
        = list(experiments_results_df["Makespan_Relaxation_Percentage"].sort_values().unique())
    target_scheduler = "MEC"
    x_ticks = [0, 1000, 2000, 3000, 4000, 5000]
    alpha = 0.6
    theme_style = "whitegrid"
    line_colors = color_palette("Set3", 1000)
    line_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    execution_parameters = {"experiment_name": experiment_name,
                            "experiments_analysis_results_folder": experiments_analysis_results_folder,
                            "experiments_results_df": experiments_results_df,
                            "num_resources": num_resources,
                            "scheduler_names": scheduler_names,
                            "metrics_names": metrics_names,
                            "makespan_relaxation_percentages": makespan_relaxation_percentages,
                            "target_scheduler": target_scheduler,
                            "x_ticks": x_ticks,
                            "alpha": alpha,
                            "theme_style": theme_style,
                            "line_colors": line_colors,
                            "line_sizes": line_sizes}
    # Generate the experiments results figures.
    generate_experiments_results_figures(execution_parameters)
    # Check how many times other schedulers met the performance of the target scheduler for a set of metrics.
    compare_schedulers_metrics_performance(execution_parameters)
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
