# Euro-Par 2024

A summary for reproducibility.

#### **Article Title:** *Optimal Time and Energy-Aware Client Selection Algorithms for Federated Learning on Heterogeneous Resources*

#### **Authors:**
- [Alan L. Nunes](https://orcid.org/0000-0002-9384-862X)
- [Cristina Boeres](https://orcid.org/0000-0002-1679-6643)
- [Lúcia M. A. Drummond](https://orcid.org/0000-0002-3831-5230)
- [Laércio Lima Pilla](https://orcid.org/0000-0003-0997-586X)

#### **Abstract:** <p align="justify">A Federated Learning (FL) system enables the training  of distinct models distributed in devices or resources, each one with its own private data. Iteratively, each device  sends its local model updates to a server that will perform a fusion to produce an improved global model. Due to the heterogeneous nature of the devices, client selection is critical to determine the overall training time of the system. Most client selection algorithms aim to maximize the number of devices that can finish their training subject to a given deadline. While selecting more clients can maximize the training accuracy of the model, the overall energy consumption can increase. We investigate and formulate two problems: firstly, the Minimal Makespan and Energy Consumption FL Schedule problem (MEC), and secondly, the Minimal Energy Consumption and Makespan FL Schedule under the Time Constraint problem (ECMTC). Both are total cost minimization problems with identical, independent, and atomic tasks assigned to heterogeneous resources with arbitrary cost functions. Without making any assumptions regarding the behavior or shape of the functions that give the execution time and the energy consumption on a resource, we provide optimal solutions with a dynamic programming algorithm with worst-case complexity in O($T^2n$) for a workload of $T$ tasks and $n$ heterogeneous resources. We propose pseudo-polynomial optimal solutions to the problems based on the previously unexplored Multiple-Choice Minimum-Cost Maximal Knapsack Packing Problem. We extend our proposed MEC and ECMTC algorithms to find the maximal training accuracy as their last objective in the optimization chain. We evaluate MEC, ECMTC, and their extensions in an extensive series of experiments using simulation that includes comparisons to other algorithms from the state of the art. Our results indicate that MEC and ECMTC provide optimal solutions with realistic execution times.</p>

### Compatibility and Basic Requirements

The code is compatible with systems that feature:

- A x86_64 processor, some GB of RAM, and some MB of storage;
- A Unix-based operating system such as Ubuntu and macOS. Windows with WSL (Windows Subsystem for Linux) should also be compatible.

The code requires:

- Shell to run the script files located in the `fl-optimal-schedulers/scripts` folder;
- Python 3 to run the base code;
- pip to install the Python packages (dependencies).

### Input

All inputs are generated automatically by the Python files located in the `fl-optimal-schedulers/experiments` folder.

### Setup

&nbsp;&nbsp;&nbsp;**1.** Clone the project (v0.2.2):

```bash
  git clone https://github.com/alan-lira/fl-optimal-schedulers/tree/v0.2.2
```

&nbsp;&nbsp;&nbsp;**2.** Go to the project directory:

```bash
  cd fl-optimal-schedulers
```

&nbsp;&nbsp;&nbsp;**3.** Install the dependencies (pip is required):

```bash
  bash scripts/run_setup.sh
```

### Executing Tests

To execute all the unit tests, run the following command:

```bash
  bash scripts/run_all_schedulers_tests.sh
```

### Executing Costs Experiments

To execute all the costs experiments, run the following command:

```bash
  bash scripts/run_all_costs_experiments.sh
```

The costs experiments results will be stored in the `fl-optimal-schedulers/experiments_results` folder.

### Executing Timing Experiments

To execute all the timing experiments, run the following command:

```bash
  bash scripts/run_all_timing_experiments.sh
```

The timing experiments results will be stored in the `fl-optimal-schedulers/experiments_results` folder.

### Executing Experiments Results Analyses

To execute all the experiments results analyses, run the following command:

```bash
  bash scripts/run_all_experiments_results_analyses.sh
```

The experiments analyses will be stored in the `fl-optimal-schedulers/experiments_results_analyses` folder.

### Executing All the Steps Above

To execute all the steps above, run the following command:

```bash
  bash scripts/run_everything.sh
```

### Experimental Platform

<u>Hardware:</u>

- Dell Precision 3571, featuring:
  - Intel Core vPro i7-12800H processor (14 cores, 20 threads, 24MB cache, 2.4GHz to 4.8GHz);
  - 32GB DDR5 RAM (1x32GB, 4800MHz);
  - M.2 2280, 1TB, PCIe NVMe Gen4 x4, Class 40 SSD.

<u>Operating System:</u>

- Ubuntu 20.04.5 LTS (kernel 5.15.0-91-generic).

<u>Software:</u>

- GNU Bourne Again SHell 5.0-6ubuntu1.2;
- pip 24.0;
- Python 3.9.6;
- Python packages: numpy 1.26.4, matplotlib 3.8.3, pandas 2.2.0, pyarrow 15.0.0, seaborn 0.13.2, and scipy 1.12.0.

### Results and Analyses

The results for each experimental set described in the article are available in the `fl-optimal-schedulers/reproducibility/Euro-Par_2024/results` folder. Moreover, their analyses are available in the `fl-optimal-schedulers/reproducibility/Euro-Par_2024/analyses` folder, which includes the figures and tables presented in the article.
