# VRU-related Scenario Mining in Waymo Open Motion Dataset

## Introduction

This repo contains a python implementation of VRU-related scenario mining for the dataset [Waymo Open Motion Dataset(WOMD)](https://waymo.com/open/data/motion).
For each data record in WOMD there are two JSON files that describe the mined scenarios.

- Waymo_#data_YYYY-MM-DD-HH_MM_**solo**.json
- Waymo_#data_YYYY-MM-DD-HH_MM_**tag**.json

## Dataset downloading

The ''tfexample'' from [WOMD](https://waymo.com/open/data/motion) is organized by time-frame.
Download dataset in shell:

```shell
gcloud init # not needed after the first time
```

```shell
gcloud auth login # go to browser and login with google account
```

```shell
gsutil config # config project name
```

```shell
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example" \
  "your_data_dir"
```

## Usage

1. Install the packages in virtual environment with [requirements.txt](requirements.txt). The code is tested with Python 3.8.10.

2. To generate the tags, you should first check [./utils/runner.py](./utils/runner.py).
    - Before running the code, please modify the data directory.

    ```python
    # current working directory
    ROOT = Path(__file__).parent.parent

    # modify the following line to your own data directory
    DATA_DIR = ROOT / "waymo_open_dataset/data/tf_example/training"
    ```

    - Run with the following command

    ```shell
    python .\utils\runner.py
    ```

    - The tags will be stored in an automatically created folder which is named by the time you run the code in [./results/gp1/](./results/gp1/)

3. For scenario categorization, run the follwoing command

```shell
python .\utils\runner_sc.py --result_time [result_time] # check the folder where the tags is stored. Format: MM-DD-HH_MM

4. **Runner_plot has no function for making plots from data and tags by now.** Don't use it. After the generation of JSON files, you can also use [./utils/runner_plot.py](./utils/runner_plot.py) to **visualize** the one-actor scenarios.
    - This time you also need to modify the directory for storing figures.

    ```python
    # modify the following two lines to your own data,figures, and result directory
    DATADIR = ROOT / "waymo_open_dataset/data/tf_example/training"
    FIGDIR = ROOT / "figures/scenarios/v3"
    RESULTDIR = ROOT / "results/v6"
    ```

    - Again, comment out the wechat messager if it does not work for you.

    ```python
    # messager for finishing one data record. Comment out this if you don't use wechat
    wechatter(f"Error in plotting {FILENUM}")
    ```

5. Logs are located at the folder [log](./utils/logger/log/).
