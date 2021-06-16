import os, subprocess
import multiprocessing as mp


def clip_data(filename):

	os.makedirs(output_dir, exist_ok=True)
	clip_cmd = ['wine', os.path.join(lastools_path, 'lasclip64.exe'), '-i', os.path.join(input_dir, filename), '-poly', shapefile, '-odir', output_dir, '-olaz']
	#print(clip_cmd)
	subprocess.run(clip_cmd)
	
	
	
	
	
   
if __name__ == "__main__":
	input_dir = '/media/external/data/SSE/Demo/A71'
	output_dir = '/media/external/data/SSE/Demo_clipped/A71'
	shapefile = '/media/external/data/SSE/Demo_corridor_shape/SpanBuffer25.shp'
	lastools_path = '/home/nmgml/wine/LasTools'
	
	with mp.Pool(mp.cpu_count()) as p:
	    p.map(clip_data, os.listdir(input_dir))
    print("Clipped data.")
