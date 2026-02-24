import os
import datetime
from time import time 
from optimizer import Optimizer

class Args:
    def __init__(self, model = "Model1", composition_file = "composition.yaml", root_output_dir = "results", dir_name = "", seed = 0, one_island = False):
        self.model = model
        self.composition_file = composition_file
        self.root_output_dir = root_output_dir
        self.dir_name = dir_name
        self.seed = seed
        self.one_island = one_island

process_composits = [
    Args(model="OneIsland",composition_file="composition2.yaml",root_output_dir="results",dir_name="OneIsland_3",seed=2,one_island=True),
    Args(model="OneIsland",composition_file="composition2.yaml",root_output_dir="results",dir_name="OneIsland_4",seed=3,one_island=True),

    #Args(model="QED",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_QED_3",seed=2),
    #Args(model="DRD2",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_DRD2_3",seed=2),
    #Args(model="GSK3B",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_GSK3B_3",seed=2),
    #Args(model="Mestranol",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_Mestranol_3",seed=2),
    #Args(model="Thiothixene",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_Thiothixene_3",seed=2),
    #Args(model="Perindopril",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_Perindorpil_3",seed=2),
    #Args(model="Isomer",composition_file="molleo2.yaml",root_output_dir="results",dir_name="base_Isomer_3",seed=2),
]#

for process_composit in process_composits:
    if not os.path.exists(process_composit.root_output_dir):
        os.mkdir(process_composit.root_output_dir)

    if process_composit.dir_name == "":
        dt = datetime.datetime.now() + datetime.timedelta(hours=9)
        process_composit.root_output_dir = os.path.join(process_composit.root_output_dir,dt.strftime('%m-%d_%H:%M'))
    else:
        process_composit.root_output_dir = os.path.join(process_composit.root_output_dir,process_composit.dir_name)

    if not os.path.exists(process_composit.root_output_dir):
        os.mkdir(process_composit.root_output_dir)
    
    start_time = time()
    
    optimizer = Optimizer(args=process_composit)
    optimizer.optimize()

    end_time = time()

    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))

    print(f"process {process_composit.model} finished !!")