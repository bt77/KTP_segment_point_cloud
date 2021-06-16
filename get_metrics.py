import util.metric as metric
import argparse, pdal, json, os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--gt", help="path to ground truth .laz files")
parser.add_argument("--pred", help="path to cleaned .laz file in postprocessing")
FLAGS = parser.parse_args()

labels_names = ["Ground","PL", "Veg", "Building", "Conductor", "Other", "Brdige", "Noise", "Water", "Road", "GuyWire"]


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
	print(f'classes in {file_path}: {set(classifications)}')
	print(f'no. points in {file_path}: {count}')
	del data

	return classifications
	
def merge_las(files_dir):
	merge_pipeline={
	"pipeline": [
		os.path.join(files_dir, '*.la*'),
		{
        		"type": "filters.merge"
    		},
		{
			"type":"filters.sample",
			"radius": 0.001
		}
	]
	}
	
	r = pdal.Pipeline(json.dumps(merge_pipeline))
	r.validate()
	count = r.execute()
	
	data = r.arrays[0]
	sort_idx = np.argsort(data)
	classifications = list(zip(*data[sort_idx]))[8]
	print(f'classes in {files_dir}: {set(classifications)}')
	print(f'no. points in {files_dir}: {count}')
	del data

	return classifications

def get_metrics():

	confusion_matrix = metric.ConfusionMatrix(len(labels_names))
	
	pred_val = read_las(FLAGS.pred)
	gt_val = list(merge_las(FLAGS.gt))
	# change class code 7 (GuyWire) in gt to num_class+1 as 7 should be noise
	gt_val = [len(labels_names)-1 if x ==7 else x for x in gt_val]
	print('gt_val updated classes:', set(gt_val))

	confusion_matrix.increment_from_list(gt_val, pred_val)
								
	acc = confusion_matrix.get_accuracy()
	mIoU = confusion_matrix.get_mean_iou()

	if mIoU != None:
		print("Average IoU : %f" % (mIoU))
	
	iou_per_class = confusion_matrix.get_per_class_ious()	
	for i in range(len(labels_names)):
			print("IoU of %s : %s" % (labels_names[i], iou_per_class[i]))
	
	confusion_matrix.print_metrics()
				
				
				
if __name__ == "__main__":
	get_metrics()
