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
    Args(model="BioT5x3",composition_file="composition.yaml",root_output_dir="results",dir_name="BioT5x3_cluster",seed=0),
    #Args(model="JNK3",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_jnk3_1",seed=0),
    #Args(model="QED",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_QED_2",seed=1),
    #Args(model="DRD2",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_DRD2_2",seed=1),
    #Args(model="GSK3B",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_GSK3B_2",seed=1),
    #Args(model="Mestranol",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_Mestranol_2",seed=1),
    #Args(model="Thiothixene",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_Thiothixene_2",seed=1),
    #Args(model="Perindopril",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_Perindopril_2",seed=1),
    #Args(model="Isomer",composition_file="composition.yaml",root_output_dir="results",dir_name="cluster_Isomer_2",seed=1),
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