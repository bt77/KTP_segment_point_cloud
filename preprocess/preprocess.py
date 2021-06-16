import os, subprocess, json, random, pdal
import multiprocessing as mp
import argparse
import pandas as pd
from functools import partial


def filter_noise(area):
	# Keep manual Noise by comparing P24 vs P30, only used for train & val data
	print(f"{area}: Filtering 7-LowPoint ...")
	# Create dataset folders
	os.makedirs(os.path.join(noise_dir, area), exist_ok=True)
	
	# Extract 7-LowPoint from P24 & P30
	with mp.Pool(2) as p:
	    func = partial(extract_lowpoint, area)
	    p.map(func, ["p24", "p30"])
	
	
	# Remove duplicates
	dedup_cmd = ['wine', os.path.join(lastools_path, 'lasduplicate.exe'), '-i', os.path.join(noise_dir, area, 'LowPoint_p24.laz'), os.path.join(noise_dir, area, 'LowPoint_p30.laz'), '-merged', '-unique_xyz', '-o', 'LowPoint_unique.laz', '-odir', os.path.join(noise_dir, area)]
	print(dedup_cmd)
	subprocess.run(dedup_cmd)
	
	# Keep manual Noise only, set unused class code
	filter_cmd = ['wine', os.path.join(lastools_path, 'las2las.exe'), '-i', os.path.join(noise_dir, area, 'LowPoint_unique.laz'), '-keep_user_data', '1', '-set_classification', temp_class, '-o', 'LowPoint_manual.laz', '-odir', os.path.join(noise_dir, area)]
	print(filter_cmd)
	subprocess.run(filter_cmd)
	print("Filtered noise: ", os.path.join(noise_dir, area, 'LowPoint_manual.laz'))
	
	
	
def extract_lowpoint(area, product):	
	if product == "p30":
	    in_dir = raw_dir
	    user_data = '1'
	    out_file = 'LowPoint_p30.laz'
	if product == "p24":
	    in_dir = p24_dir
	    user_data = '0'
	    out_file = "LowPoint_p24.laz"
		
	filter_cmd = ['wine', os.path.join(lastools_path, 'las2las.exe'), '-i', os.path.join(in_dir, area, '*.la*'), '-keep_class', '7', '-merged', '-set_user_data', user_data, '-o', out_file, '-odir', os.path.join(noise_dir, area)]
	print(filter_cmd)
	subprocess.run(filter_cmd)


def filter_overlap(area):
	# Keep 12-Overlap of P24 within specified HAG range
	print(f"{area}: Filtering 12-Overlap ...")
	# Create dataset folders
	os.makedirs(os.path.join(overlap_dir, area), exist_ok=True)
	os.makedirs(os.path.join(overlap_filtered_dir, area), exist_ok=True)
	
	hag_cmd = ['wine', os.path.join(lastools_path, 'lasheight.exe'), '-i', os.path.join(raw_dir, area, '*.la*'), '-keep_class', '12', '-classify_between', str(overlap_hag_range[0]), str(overlap_hag_range[1]), temp_class, '-olaz', '-odir', os.path.join(overlap_dir, area), '-cores', str(mp.cpu_count())]
	print(hag_cmd)
	subprocess.run(hag_cmd)
	
	filter_cmd = ['wine', os.path.join(lastools_path, 'las2las.exe'), '-i', os.path.join(overlap_dir, area, '*.laz'), '-keep_class', temp_class, '-olaz', '-odir', os.path.join(overlap_filtered_dir, area), '-cores', str(mp.cpu_count())]
	print(filter_cmd)
	subprocess.run(filter_cmd)

def tile_area(area, filtered_input):
	# Create dataset folders
	os.makedirs(os.path.join(tiled_dir, area), exist_ok=True)
		
	# Following steps happen during the tiling:
	# 1. filter classes
	# 2. merge input tiles
	# 3. rescale intensity
	# 4. tile with buffer
	print(f'{area}: Tiling ...')
	filter_cmd = [v for elt in drop_class for v in ('-drop_class', str(elt))]
	tile_cmd = ['wine', os.path.join(lastools_path, 'lastile64.exe'), '-i', os.path.join(raw_dir, area, '*.la*'), filtered_input, '-scale_intensity', intensity_scale_factor, '-merged', '-tile_size', tile_length, '-buffer', tile_buffer, '-olaz', '-odir', os.path.join(tiled_dir, area)]
	tile_cmd.extend(filter_cmd)
	print(tile_cmd)
	subprocess.run(tile_cmd)
	print('Tiled: ', os.path.join(tiled_dir, area))

def merge_pc_class(area, pc):
    in_file = os.path.join(tiled_dir, area, pc)
    out_file = os.path.join(merged_dir, area, pc)
    subprocess.run(["pdal", "pipeline", pdal_config, f"--readers.las.filename={in_file}", f"--writers.las.filename={out_file}"])
    	
def merge_class(area):
	print(f'{area}: Merging classes ...')
	# Create dataset folders
	os.makedirs(os.path.join(merged_dir, area), exist_ok=True)
	
	with mp.Pool(mp.cpu_count()) as p:
		func = partial(merge_pc_class, area)
		p.map(func, os.listdir(os.path.join(tiled_dir, area)))
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
	elif split: 	# training data
		if set(data['Classification']) & set(PARAMS["keep_all"]):  # Keep all tiles w/ specified class  
			keep = 1
		else:    
			keep = random.choices([0,1], weights=train_prob_essential)[0]     # Keep tiles w/o specified class with specified probs
	else:	# validation data
		#keep = 1	# For Tx, keep all validation tiles
		# For Dx, keep essential tiles & some of non-essentional tiles to reduce inaccurate GT
		if set(data['Classification']) & set(PARAMS["keep_all"]):  # Keep all tiles w/ specified class
			keep = 1
		else:    
			keep = random.choices([0,1], weights=val_prob_essential)[0]     # Keep tiles w/o specified class with specified probs
	return keep

def split_pc(in_dir_for_split, header, pc):
    file_name = pc[:-4]
	# Load points
    data = read_las(os.path.join(in_dir_for_split, pc), header)
    
    if data_split == "train":
        # Create dataset folders
        train_dir = os.path.join(point_dir, 'train')
        validation_dir = os.path.join(point_dir, 'validation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
		
        # Assign split tag of train/validation with specified probs
        split = random.choices([1,0], weights=prob_data_split)[0]	# 1 - train; 0 - validation:
	    # Balance tiles
        keep = keep_tile(data, split)
                
        if keep:
            if split:   # train data
                data[header[:-1]].to_csv(os.path.join(train_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
                data[['Classification']].to_csv(os.path.join(train_dir, file_name + '.labels'), index=False, header=False)
            else:
                data[header[:-1]].to_csv(os.path.join(validation_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
                data[['Classification']].to_csv(os.path.join(validation_dir, file_name + '.labels'), index=False, header=False)
    else:
        # Create dataset folders
        test_dir = os.path.join(point_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        if data.shape[0] > 1:  # Only keep tiles w/ #point>1
                # Split label & points
            data[header[:-1]].to_csv(os.path.join(test_dir, file_name + '.txt'), index=False, header=False, float_format='%.3f')
            if data_split == "validation":      # Keep GT for evaluation
                data[['Classification']].to_csv(os.path.join(point_dir, file_name + '.labels'), index=False, header=False)	   
        
         

def split_dataset(in_dir_for_split):
	
	os.makedirs(point_dir, exist_ok=True)
	header=["X","Y","Z","ReturnNumber","NumberOfReturns","Intensity","UserData","Classification"]	# UserData: HeightAboveGround
	
	with mp.Pool(mp.cpu_count()) as p:
		func = partial(split_pc, in_dir_for_split, header)
		p.map(func, os.listdir(in_dir_for_split))
	
	print('Splited points & labels.')
	

def preprocess_project():
	for area in os.listdir(raw_dir):
		# Filter 7-LowPoint
		if data_split == "train" and keep_noise:
	    		print("FilterNoise On.")
	    		filter_noise(area)
	    		filtered_input = os.path.join(noise_dir, area, 'LowPoint_manual.laz')
	    	
	    	# Filter 12-Overlap	
		if data_split == "test" and keep_overlap:
	    		print("FilterOverlap On.")
	    		filter_overlap(area)
	    		filtered_input = os.path.join(overlap_filtered_dir, area, '*.laz')

		

		# Tiling
		tile_area(area, filtered_input)
	
		# Merge classes
		if data_split != "test":
			merge_class(area)
			in_dir_for_split = os.path.join(merged_dir, area)  
		else:
			print("Test data, no class merge.")
			in_dir_for_split = os.path.join(tiled_dir, area)
	    
		# Split dataset
		print(f'{area}: Splitting points & labels ...')
		split_dataset(in_dir_for_split)
	print(f'Done preprocessing. Model inputs: {point_dir}')
	
	
	
   
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", default="config.json", help="path to config file for preprocessing")
	FLAGS = parser.parse_args()
	PARAMS = json.loads(open(FLAGS.config_file).read())
	
	raw_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"])
	
	p24_dir = raw_dir + '_p24'
	noise_dir = raw_dir + '_noise'
	keep_noise = PARAMS["keep_noise"]
	temp_class = PARAMS["temp_class"]
	
	overlap_dir = raw_dir + '_overlap'
	keep_overlap = PARAMS["keep_overlap"]
	overlap_hag_range = PARAMS["overlap_hag_range"]
	overlap_filtered_dir = overlap_dir + '_filtered'
	
	
	tiled_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_tiled')
	drop_class = PARAMS["drop_class"]
	lastools_path = PARAMS["lastools_path"]
	intensity_scale_factor = str(1 / PARAMS["devide_intensity_by"])
	tile_length = PARAMS["tile_length"]
	tile_buffer = PARAMS["tile_buffer"]
	
	merged_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_merged')
	pdal_config = PARAMS["pdal_config"]
	
	point_dir = os.path.join(PARAMS["base_dir"], PARAMS["project"] + '_points')
	data_split = PARAMS["data_split"]		# train or validation or test
	prob_data_split = PARAMS["prob_data_split"]
	train_prob_essential = PARAMS["train_prob_essential"]
	val_prob_essential = PARAMS["val_prob_essential"]
	

	preprocess_project()
	
