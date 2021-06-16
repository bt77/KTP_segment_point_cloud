import argparse, json, pdal, subprocess, glob
import os
import numpy as np
import open3d
import time
import multiprocessing as mp
from functools import partial

from util.metric import ConfusionMatrix
from util.point_cloud_util import load_labels, write_labels
from pprint import pprint


                
def interpolate_labels(sparse_pc, dense_points, dense_labels, k=3):
	interpolate_pipeline={
		"pipeline": [
			dense_points,
			{
				"type":"filters.neighborclassifier",
				"k": k,
				"candidate": sparse_pc
			},
			dense_labels
		]
		}
	
	r = pdal.Pipeline(json.dumps(interpolate_pipeline))
	r.validate()
	r.execute()

	print(f'dense_labels written to {dense_labels}')

def prepare_dense_points(dense_points_path, dense_pc_path):
	data = np.loadtxt(dense_points_path, usecols=(0,1,2), delimiter=',')
	dense_pc = np.hstack((data, np.zeros((data.shape[0], 1))))
	np.savetxt(dense_pc_path, dense_pc, fmt=['%.3f','%.3f', '%.3f','%d'], header="X Y Z Classification", comments="")
	print(f'dense_pc written to {dense_pc_path}')
	
def read_las(file_path):
	pipeline={
	"pipeline": [
		{
			"type": "readers.las",
			"filename": file_path
		}
	]
	}
	
	r = pdal.Pipeline(json.dumps(pipeline))
	r.validate()
	count = r.execute()
	
	data = r.arrays[0]
	sort_idx = np.argsort(data)
	classifications = list(zip(*data[sort_idx]))[8]
	del data

	return classifications
	
def process_test_data(data_split, sparse_dataset, dense_dir, gt_dir, sparse_points_path):
        
    file_prefix = os.path.basename(sparse_points_path)[:-4]
    # Paths
    sparse_labels_path = os.path.join(sparse_dataset, file_prefix + ".labels")
    dense_points_path = os.path.join(gt_dir, file_prefix + ".txt")
    dense_pc_path = os.path.join(dense_dir, file_prefix + ".txt")
                        
    dense_labels_path = os.path.join(dense_dir, file_prefix + ".laz")
    sparse_pc_path = os.path.join(sparse_dataset, file_prefix + ".txt")
    os.makedirs(dense_dir, exist_ok=True)

    # Sparse points
    sparse_pcd = open3d.io.read_point_cloud(sparse_points_path)
    sparse_points = np.asarray(sparse_pcd.points)
    del sparse_pcd
                        
    # Sparse labels
    sparse_labels = load_labels(sparse_labels_path)
                        
    # Combine points & labels for processed points
    sparse_pc = np.column_stack((sparse_points, sparse_labels))
    np.savetxt(sparse_pc_path, sparse_pc, fmt=['%.3f','%.3f', '%.3f','%d'], header="X Y Z Classification", comments="")
    print("sparse_pc written to ", sparse_pc_path)
                        
    # Add Classification 0 to dense_points
    prepare_dense_points(dense_points_path, dense_pc_path)
                        

	# Interpolation
    interpolate_labels(sparse_pc_path, dense_pc_path, dense_labels_path)


if __name__ == "__main__":
        # Parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", default="test.json", help="Path to config file, default is test.json")
        parser.add_argument("--set", default="validation", help="Data split to process, default is validation")
        parser.add_argument("--gt", help="Path to ground truth laz files")
        
        flags = parser.parse_args()
        hyper_params = json.loads(open(flags.config_file).read())
        
        # Directories
        dataset = hyper_params["output_dir"]
        sparse_dataset = os.path.join("result", "sparse", dataset, flags.set)
        dense_dir = os.path.join("result", "dense", dataset, flags.set)
        gt_dir = os.path.join(hyper_params["data_path"], flags.set)
        
        
        if flags.set == "test":
            input_paths = glob.glob(os.path.join(sparse_dataset, '*.pcd'))
            with mp.Pool(mp.cpu_count()) as p:
                func = partial(process_test_data, flags.set, sparse_dataset, dense_dir, gt_dir)
                p.map(func, input_paths)
        else:
            # Global statistics
            cm_global = ConfusionMatrix(hyper_params["num_classes"])

            for file in os.listdir(sparse_dataset):
                    if file.endswith('.pcd'):
                            file_prefix = file[:-4]
                            # Paths
                            sparse_points_path = os.path.join(sparse_dataset, file_prefix + ".pcd")
                            sparse_labels_path = os.path.join(sparse_dataset, file_prefix + ".labels")
                            dense_points_path = os.path.join(gt_dir, file_prefix + ".txt")
                            dense_pc_path = os.path.join(dense_dir, file_prefix + ".txt")
                        
                            dense_labels_path = os.path.join(dense_dir, file_prefix + ".laz")
                            sparse_pc_path = os.path.join(sparse_dataset, file_prefix + ".txt")
                            os.makedirs(dense_dir, exist_ok=True)

                            # Sparse points
                            sparse_pcd = open3d.io.read_point_cloud(sparse_points_path)
                            sparse_points = np.asarray(sparse_pcd.points)
                            del sparse_pcd
                        
                            # Sparse labels
                            sparse_labels = load_labels(sparse_labels_path)
                        
                            # Combine points & labels for processed points
                            sparse_pc = np.column_stack((sparse_points, sparse_labels))
                            np.savetxt(sparse_pc_path, sparse_pc, fmt=['%.3f','%.3f', '%.3f','%d'], header="X Y Z Classification", comments="")
                            print("sparse_pc written to ", sparse_pc_path)
                        
                            # Add Classification 0 to dense_points
                            prepare_dense_points(dense_points_path, dense_pc_path)
                        

			                # Interpolation
                            interpolate_labels(sparse_pc_path, dense_pc_path, dense_labels_path)
                        
                            dense_gt_pc_path = os.path.join(flags.gt, file_prefix + ".laz")
                            # Eval
                            cm = ConfusionMatrix(hyper_params["num_classes"])
                            pred_val = read_las(dense_labels_path)
                            gt_val = list(read_las(dense_gt_pc_path))
                            cm.increment_from_list(gt_val, pred_val)
                            cm_global.increment_from_list(gt_val, pred_val)


            pprint("Global results")
            cm_global.print_metrics()
