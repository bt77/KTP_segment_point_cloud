import os, subprocess, pdal, json, glob
import multiprocessing as mp
import argparse


	

def update_classification(pc):
	pc_name = os.path.basename(pc)
	print(f'Updating classification: {pc}')
	point_file = os.path.join(point_dir, pc_name)
	#print(f'Point file: {point_file}')
	out_file = os.path.join(labeled_dir, pc_name)
	
	update_pipeline={
		"pipeline": [
			{
				"type": "readers.las",
				"filename": point_file
			},
			{
        			"type":"filters.python",
        			"script":"replace_class.py",
        			"function":"replace_class",
        			"module":"anything",
				"pdalargs":"{\"label_file\": \"" + pc + "\"}"
			},
			{
        			"type":"writers.las",
        			"filename": out_file,
				"forward": "vlr"
    			}
		]
		}
	
	r = pdal.Pipeline(json.dumps(update_pipeline))
	r.validate()
	r.execute()
	


def process_dataset():
	print(f'Processing {in_dir}')
	
	input_paths = glob.glob(os.path.join(in_dir, '*.laz'))
	with mp.Pool(mp.cpu_count()) as p:
		p.map(update_classification, input_paths)
	print('Updated classification:', labeled_dir)

	# Remove duplicated points & rescale intensity 
	crop_cmd = ['wine', os.path.join(lastools_path, 'lastile64.exe'), '-i', labeled_dir + '/*.laz', '-remove_buffer', '-olaz', '-odir', cropped_dir, '-scale_intensity', intensity_scale_factor]
	subprocess.run(crop_cmd)
	print('Cropped area: ', os.path.join(cropped_dir))
	
	
	# Retile PC
	tile_cmd = ['wine', os.path.join(lastools_path, 'lastile64.exe'), '-i', cropped_dir + '/*.laz', '-olaz', '-odir', output_dir]
	subprocess.run(tile_cmd)
	print('Final outputs: ', output_dir)

		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", help="path to config file for postprocessing")

	FLAGS = parser.parse_args()
	PARAMS = json.loads(open(FLAGS.config_file).read())
	in_dir = PARAMS["in_dir"]
	point_dir = PARAMS["point_dir"]		# Folder containing original points
	
	out_dir = PARAMS["out_dir"]
	labeled_dir = out_dir + '_labeled'	# Folder containing pc with predicted labels
	lastools_path = PARAMS["lastools_path"]
	cropped_dir = out_dir + '_cropped'	# Folder containing the cropped .laz for an area
	cleaned_dir = out_dir + '_cleaned'	# Folder containing the merged & cleaned .laz for an area
	output_dir = out_dir + '_final'		# Folder containing the final outputs
	
	intensity_scale_factor = PARAMS["multiply_intensity_by"]	# Rescale intensity according to devide_intensity_by in preprocessing
	
	
	# Create dataset dirs
	os.makedirs(labeled_dir, exist_ok=True)
	os.makedirs(cropped_dir, exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)
	
	process_dataset()
	
