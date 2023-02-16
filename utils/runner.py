"""
Generate scenarios from a folder contains multiple WAYMO data records
Author: Detian Guo
Date: 04/11/2022
"""
import os
import time
import re
import json
import argparse
from TagsGenerator import TagsGenerator
from ScenarioMiner import ScenarioMiner
from rich.progress import track
from logger.logger import *
import traceback
from helpers.wechatter import wechatter
from parameters import *

# working directory 
# reslove() is to get the absolute path
ROOT = Path(__file__).resolve().parent.parent


# modify the following two lines to your own data and result directory
DATADIR = ROOT / "waymo_open_dataset/data/tf_example/training"

DATADIR_WALK = DATADIR.iterdir()
RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M",time.localtime())
RESULTDIR = ROOT / "results/gp1" / RESULT_TIME
if not RESULTDIR.exists():
    RESULTDIR.mkdir()

# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, required=True, help='#file to plot.e.g.:00003')
# args = parser.parse_args()

if __name__ == '__main__':
    time_start = time.perf_counter()
    for DATA_PATH in track(DATADIR_WALK,description="Processing files"):
        FILE = DATA_PATH.name
        FILENUM = re.search(r"-(\d{5})-",FILE)
        if FILENUM is not None:
            FILENUM = FILENUM.group()[1:-1]
            print(f"Processing file: {FILE}")
        else:
            print(f"File name error: {FILE}")
            continue
        result_dict = {}
        RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_tag.json'
        try:
            tags_generator = TagsGenerator()
            actors_list,\
            inter_actor_relation,\
            actors_activity,\
            actors_environment_element_intersection = tags_generator.tagging(DATADIR,FILE)
            result_dict = {
            'actors_list':actors_list,
            'inter_actor_relation':inter_actor_relation,
            'actors_activity':actors_activity,
            'actors_environment_element_intersection':actors_environment_element_intersection
            }
            with open(RESULTDIR / RESULT_FILENAME,'w') as f:
                print(f"Saving tags.")
                json.dump(result_dict,f)
            scenario_miner = ScenarioMiner()
            solo_scenarios = scenario_miner.mining(result_dict)
            RESULT_FILENAME = f'Waymo_{FILENUM}_{RESULT_TIME}_solo.json'
            with open(RESULTDIR / RESULT_FILENAME,'w') as f:
                print(f"Saving solo scenarios.")
                json.dump(solo_scenarios,f)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"FILE:{FILENUM}.\nTag generagtion:{e}")
            logger.error(f"trace:{trace}")
    ############################################################################
        # messager for finishing one data record. Comment out this if you don't use wechat
        # wechatter(f"FILE:{FILENUM} finished.")
    ############################################################################
    time_end = time.perf_counter()
    print(f"Time cost: {time_end-time_start:.2f}s.RESULTDIR: {RESULTDIR}")
    logger.info(f"Time cost: {time_end-time_start:.2f}s.RESULTDIR: {RESULTDIR}")

