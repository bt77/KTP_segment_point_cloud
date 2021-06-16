import shutil, os
import multiprocessing as mp
import sys, random
import pandas as pd
import numpy as np



in_dir = sys.argv[1]
out_dir = sys.argv[2]
keep_classes = [1, 4, 5]


def keep_tile(data):
	if set(data) & set(keep_classes):  # Keep all tiles w/ specified class	
		keep = 1
	else:	 
		keep = random.choices([0,1], weights=[0.9, 0.1])[0]	# Keep 10% tiles w/o specified class
	return keep


def remove_files():
	for file in os.listdir(in_dir):
		if file.endswith('.labels'):
			file_in = os.path.join(in_dir, file)
			data = np.loadtxt(file_in)
			file_name = os.path.basename(file)[:-7]
		
			keep = keep_tile(data)
	
			if keep == 0:
				shutil.move(file_in, os.path.join(out_dir, file))
				shutil.move(os.path.join(in_dir, file_name + '.txt'), os.path.join(out_dir, file_name + '.txt'))
				print(f'Removed {file}')



if __name__ == "__main__":
	random.seed(1)
	remove_files()
