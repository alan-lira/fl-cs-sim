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
    experiment_results_analysis_folder = execution_parameters["experiment_results_analysis_folder"]
    experiment_results_df = execution_parameters["experiment_results_df"]
    num_resources = execution_parameters["num_resources"]
    scheduler_names = execution_parameters["scheduler_names"]
    metrics_names = execution_parameters["metrics_names"]
    x_ticks = execution_parameters["x_ticks"]
    alpha = execution_parameters["alpha"]
    theme_style = execution_parameters["theme_style"]
    line_colors = execution_parameters["line_colors"]
    line_sizes = execution_parameters["line_sizes"]
    # Set the visual theme for all matplotlib and seaborn plots.
    set_theme(style=theme_style)
    # Set the matplotlib color cycle using a seaborn palette.
    colors = color_palette(line_colors)
    set_palette(colors, len(scheduler_names))
    # Generate a figure for each (num_resources, metric_name) tuple.
    for n_resources in num_resources:
        data = experiment_results_df[experiment_results_df["Num_Resources"] == n_resources]
        for metric_name in metrics_names:
            if metric_name == "Num_Selected_Resources":
                y_label = metric_name.replace("_", " ").replace("Num", "Number of").capitalize()
            else:
                y_label = metric_name.replace("_", " ").capitalize()
                if y_label == "Makespan":
                    y_label = "Makespan (s)"
                elif y_label == "Energy consumption":
                    y_label = "Energy consumption (J)"
                elif y_label == "Training accuracy":
                    y_label = "Weighted mean accuracy"
            figure(figsize=(6, 5))
            rcParams["axes.titlesize"] = 18
            rcParams["axes.labelsize"] = 18
            rcParams["xtick.labelsize"] = 18
            rcParams["ytick.labelsize"] = 18
            rcParams["legend.fontsize"] = 16
            xlabel("Number of tasks", fontsize=18)
            ylabel(y_label, fontsize=18)
            xticks(ticks=x_ticks, rotation=15)
            ax = lineplot(data=data,
                          x="Num_Tasks",
                          y=metric_name,
                          hue="Scheduler_Name",
                          hue_order=scheduler_names,
                          style="Scheduler_Name",
                          dashes=False,
                          markers=True,
                          alpha=alpha,
                          size="Scheduler_Name",
                          sizes=line_sizes,
                          markersize=6)
            move_legend(ax,
                        "lower center",
                        bbox_to_anchor=(.5, 1),
                        ncol=3,
                        title=None,
                        frameon=True)
            output_figure_file = Path("fig_{0}_{1}_resources_{2}.pdf".format(experiment_name, n_resources, metric_name.lower()))
            output_figure_file_full_path = experiment_results_analysis_folder.joinpath(output_figure_file)
            savefig(output_figure_file_full_path, bbox_inches="tight")
            print("Figure '{0}' was successfully generated.".format(output_figure_file_full_path))


def compare_schedulers_metrics_performance(execution_parameters: dict) -> None:
    experiment_results_df = execution_parameters["experiment_results_df"]
    scheduler_names = execution_parameters["scheduler_names"]
    metrics_names = execution_parameters["metrics_names"]
    target_schedulers = execution_parameters["target_schedulers"]
    for target_scheduler in target_schedulers:
        print("\nSchedulers metrics performance comparison (target scheduler: '{0}'):\n".format(target_scheduler))
        for metric_name in metrics_names:
            target_scheduler_cost = experiment_results_df[experiment_results_df["Scheduler_Name"] ==
                                                          target_scheduler][metric_name].reset_index(drop=True)
            for scheduler in scheduler_names:
                if scheduler != target_scheduler:
                    other_cost = experiment_results_df[experiment_results_df["Scheduler_Name"] ==
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
    experiment_name = "random_costs"
    # Start message.
    print("{0}: Starting the '{1}' experiment's analysis...".format(datetime.now(), experiment_name))
    # Get the experiments results folder.
    experiments_results_folder = Path(__file__).resolve().parents[1].joinpath("experiments_results")
    # Get the experiment results CSV file.
    experiment_results_csv_file_name = "{0}_experiment_results.csv".format(experiment_name)
    experiment_results_csv_file = experiments_results_folder.joinpath(experiment_results_csv_file_name)
    # Set the experiments results analyses folder.
    experiments_results_analyses_folder = Path(__file__).resolve().parents[1].joinpath("experiments_results_analyses")
    # Set the experiment results analysis folder.
    experiment_results_analysis_folder = experiments_results_analyses_folder.joinpath(experiment_name)
    # Remove the output folder and its contents (if exists).
    if experiment_results_analysis_folder.is_dir():
        rmtree(experiment_results_analysis_folder)
    # Create the parents directories of the output file (if not exist yet).
    experiment_results_analysis_folder.parent.mkdir(exist_ok=True, parents=True)
    # Create the output folder.
    experiment_results_analysis_folder.mkdir(exist_ok=True, parents=True)
    # Load the dataframe from the experiment results CSV file.
    experiment_results_df = read_csv(experiment_results_csv_file, comment="#")
    # Sort the dataframe in ascending order of the schedulers names.
    experiment_results_df = experiment_results_df.sort_values(by=["Scheduler_Name"], ascending=True)
    # Set the execution parameters.
    num_resources = list(experiment_results_df["Num_Resources"].sort_values().unique())
    scheduler_names = list(experiment_results_df["Scheduler_Name"].sort_values().unique())
    metrics_names = experiment_results_df.columns.drop(["Scheduler_Name", "Num_Tasks", "Num_Resources"]).to_list()
    target_schedulers = ["MEC", "ECMTC"]
    x_ticks = [0, 1000, 2000, 3000, 4000, 5000]
    alpha = 0.6
    theme_style = "whitegrid"
    line_colors = ["#00FFFF", "#FFA500", "#E0115F", "#0000FF", "#7FFFD4", "#228B22"]
    line_sizes = [6, 2, 2, 2, 6, 2]
    execution_parameters = {"experiment_name": experiment_name,
                            "experiment_results_analysis_folder": experiment_results_analysis_folder,
                            "experiment_results_df": experiment_results_df,
                            "num_resources": num_resources,
                            "scheduler_names": scheduler_names,
                            "metrics_names": metrics_names,
                            "target_schedulers": target_schedulers,
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
