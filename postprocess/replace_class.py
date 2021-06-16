import pdal, json
import numpy as np

def replace_class(ins,outs):
	# Convert dense pc from .laz to .txt
	pipeline={
	"pipeline": [
		{
			"type": "readers.las",
			"filename": pdalargs["label_file"]
		},
		{
        		"type":"writers.text",
        		"order":"Classification",
        		"keep_unspecified":"false",
        		"filename": pdalargs["label_file"][:-3] + 'txt',
        		"write_header": "false"
    		}
	]
	}
	
	r = pdal.Pipeline(json.dumps(pipeline))
	r.validate()
	r.execute()
	
	#print('gt labels:', ins['Classification'])
	labels = np.loadtxt(pdalargs["label_file"][:-3] + 'txt', dtype=np.ubyte)
	#print('predicted labels:', labels)
	
	# Replace original labels with predicted labels
	outs['Classification'] = labels
	#print('out labels:', outs['Classification'])
	del r
	return True
