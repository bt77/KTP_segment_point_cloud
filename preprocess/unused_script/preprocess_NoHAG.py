import os, subprocess, json, random, pdal
import multiprocessing as mp
import argparse
import pandas as pd



def tile_area(area):
	# Create dataset folders
	os.makedirs(os.path.join(tiled_dir, area), exist_ok=True)
		
	# Following steps happen during the tiling:
	# 1. filter classes
	# 2. merge all files
	# 3. rescale intensity
	# 4. tile with buffer
	print(f'{area}: Tiling ...')
	filter_cmd = [v for elt in drop_class for v in ('-drop_class', str(elt))]
	tile_cmd = ['wine', os.path.join(lastools_path, 'lastile64.exe'), '-i', os.path.join(raw_dir, area + '/*.la*'),  '-merged', '-scale_intensity', intensity_scale_factor, '-tile_size', tile_length, '-buffer', tile_buffer, '-olaz', '-odir', os.path.join(tiled_dir, area)]
	tile_cmd.extend(filter_cmd)
	#print(tile_cmd)
	subprocess.run(tile_cmd)
	print('Tiled: ', os.path.join(tiled_dir, area))
	
def merge_class(area):
	print(f'{area}: Merging classes ...')
	# Create dataset folders
	os.makedirs(os.path.join(merged_dir, area), exist_ok=True)
	
	for pc in os.listdir(os.path.join(tiled_dir, area)):
		in_file = os.path.join(tiled_dir, area, pc)
		out_file = os.path.join(merged_dir, area, pc)
	
		subprocess.run(["pdal", "pipeline", pdal_config, f"--readers.las.filename={in_file}", f"--writers.las.filename={out_file}"])
	print('Merged classes:', os.path.join(merged_dir, area))

def read_las(file_path, header):
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
	r.execute()
	
	data = pd.DataFrame(r.arrays[0])[header]
	del r
	return data
		
def keep_tile(data, split):
	if data.shape[0] == 1:  # Remove tiles w/ #point=1
		keep = 0
	elif split:
		if set(data['Classification']) & set(PARAMS["keep_all"]):  # Keep all tiles w/ specified class  
			keep = 1
		else:    
			keep = random.choices([0,1], weights=prob_essential)[0]     # Keep tiles w/o specified class with specified probs
	else:
		keep = 1
	return keep

def split_dataset(area):
	print(f'{area}: Splitting points & labels ...')
	
	os.makedirs(point_dir, exist_ok=True)
	header=["X","Y","Z","ReturnNumber","NumberOfReturns","Intensity","Classification"]	# UserData: HeightAboveGround
	
	if training:
		# Create dataset folders
		train_dir = os.path.join(point_dir, 'train')
		validation_dir = os.path.join(point_dir, 'validation')
		os.makedirs(train_dir, exist_ok=True)
		os.makedirs(validation_dir, exist_ok=True)
	
		for pc in os.listdir(os.path.join(merged_dir, area)):
			# Assign split tag of train/validation with specified probs
			split = random.choices([1,0], weights=prob_data_split)[0]	# 1 - train; 0 - validation:
			# Load points
			data = read_las(os.path.join(merged_dir, area, pc), header)
			keep = keep_tile(data, split)
			file_name = pc[:-4]
                
			if keep:
				if split:
					data[header[:-1]].to_csv(os.path.join(train_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
					data[['Classification']].to_csv(os.path.join(train_dir, file_name + '.labels'), index=False, header=False)
				else:
					data[header[:-1]].to_csv(os.path.join(validation_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
					data[['Classification']].to_csv(os.path.join(validation_dir, file_name + '.labels'), index=False, header=False)
	else:
		print("Inference-only data, no balancing performed.")
		for pc in os.listdir(os.path.join(merged_dir, area)):
			file_name = pc[:-4]
			# Load points
			data = read_las(os.path.join(merged_dir, area, pc), header)
			# Split label & points
			data[header[:-1]].to_csv(os.path.join(point_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
			data[['Classification']].to_csv(os.path.join(point_dir, file_name + '.labels'), index=False, header=False)
	print(f'{area}: Splited points & labels.')
	

def preprocess_project(p):
	# Tiling
	#p.map(tile_area, os.listdir(raw_dir))
	
	# Feature engineering / Filtering
	#p.map(merge_class, os.listdir(tiled_dir))
	
	# Split dataset
	p.map(split_dataset, os.listdir(merged_dir))
	print(f'Done preprocessing. Model inputs: {point_dir}')
	
	
	
   
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", default="config.json", help="path to config file for preprocessing")
	FLAGS = parser.parse_args()
	PARAMS = json.loads(open(FLAGS.config_file).read())
	
	num_cores = PARAMS["num_cores"]
	
	raw_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"])

	tiled_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_tiled')
	drop_class = PARAMS["drop_class"]
	lastools_path = PARAMS["lastools_path"]
	intensity_scale_factor = str(1 / PARAMS["devide_intensity_by"])
	tile_length = PARAMS["tile_length"]
	tile_buffer = PARAMS["tile_buffer"]
	
	merged_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_merged')
	pdal_config = PARAMS["pdal_config"]
	
	point_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_points')
	training = PARAMS["training"]		# training or inference-only
	prob_data_split = PARAMS["prob_data_split"]
	prob_essential = PARAMS["prob_essential"]

	p = mp.Pool(num_cores)
	preprocess_project(p)
	
