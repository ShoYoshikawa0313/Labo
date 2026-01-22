import os
import datetime
from time import time 
from optimizer import Optimizer

class Args:
    def __init__(self, model = "Model1", composition_file = "composition.yaml", root_output_dir = "results", dir_name = "", seed = 0, resume = ""):
        self.model = model
        self.composition_file = composition_file
        self.root_output_dir = root_output_dir
        self.dir_name = dir_name
        self.seed = seed
        self.resume = resume

process_composits = [
    Args(model="JNK3",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_jnk3_1",seed=0),
    Args(model="JNK3",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_jnk3_2",seed=1),
    #Args(model="QED",composition_file="composition.yaml",root_output_dir="results",dir_name="QED_1",seed=0),
    #Args(model="DRD2",composition_file="composition.yaml",root_output_dir="results",dir_name="DRD2_1",seed=0),
    #Args(model="GSK3B",composition_file="composition.yaml",root_output_dir="results",dir_name="GSK3B_1",seed=0),
    #Args(model="Mestranol",composition_file="composition.yaml",root_output_dir="results",dir_name="Mestranol_1",seed=0),
    #Args(model="Thiothixene",composition_file="composition.yaml",root_output_dir="results",dir_name="Thiothixene_1",seed=0),
    #Args(model="Perindopril",composition_file="composition.yaml",root_output_dir="results",dir_name="Perindopril_1",seed=0),
    #Args(model="Isomer",composition_file="composition.yaml",root_output_dir="results",dir_name="Isomer_1",seed=0),
]

for process_composit in process_composits:
    if not os.path.exists(process_composit.root_output_dir):
        os.mkdir(process_composit.root_output_dir)

    if process_composit.resume != "":
        process_composit.root_output_dir = process_composit.resume
    elif process_composit.dir_name == "":
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