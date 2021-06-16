import sys, json
import tensorflow as tf
import argparse
import model

# Two global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="path to saved model")
parser.add_argument("--config_file", default="test.json", help="config file path")

FLAGS = parser.parse_args()
PARAMS = json.loads(open(FLAGS.config_file).read())

def get_learning_rate(batch):
        """Compute the learning rate for a given batch size and global parameters

        Args:
                batch (tf.Variable): the batch size

        Returns:
                scalar tf.Tensor: the decayed learning rate
        """

        learning_rate = tf.train.exponential_decay(
                PARAMS["learning_rate"],  # Base learning rate.
                batch * PARAMS["batch_size"],  # Current index into the dataset.
                PARAMS["decay_step"],  # Decay step.
                PARAMS["learning_rate_decay_rate"],  # Decay rate.
                staircase=True,
        )
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        return learning_rate

def get_bn_decay(batch):
        """Compute the batch normalisation exponential decay

        Args:
                batch (tf.Variable): the batch size

        Returns:
                scalar tf.Tensor: the batch norm decay
        """

        bn_momentum = tf.train.exponential_decay(
                PARAMS["bn_init_decay"],
                batch * PARAMS["batch_size"],
                float(PARAMS["decay_step"]),
                PARAMS["bn_decay_decay_rate"],
                staircase=True,
        )
        bn_decay = tf.minimum(PARAMS["bn_decay_clip"], 1 - bn_momentum)
        return bn_decay
        
def count_num_params():
	"""Train the model on a single GPU
	"""
	with tf.Graph().as_default():
		
		with tf.device("/gpu:0"):
			pointclouds_pl, labels_pl, smpws_pl = model.get_placeholders(PARAMS["num_point"], hyperparams=PARAMS)
			is_training_pl = tf.placeholder(tf.bool, shape=())

			batch = tf.Variable(0)
			bn_decay = get_bn_decay(batch)
			tf.summary.scalar("bn_decay", bn_decay)

			print("--- Get model and loss")
			# Get model and loss
			pred, end_points = model.get_model(pointclouds_pl,is_training_pl,PARAMS["num_classes"],hyperparams=PARAMS,bn_decay=bn_decay,)
			if PARAMS["loss_func"] == "focal":
				if PARAMS["balancing_factor"]:
					print("Using default focal loss")
					loss = tf.reduce_mean(model.focal_loss_softmax(labels=labels_pl, logits=pred, gamma=PARAMS["focusing_param"], alpha=PARAMS["balancing_factor"])) # focal loss weighted by specified balancing factor
				else:
					print("Using focal loss weighted by class")
					loss = tf.reduce_mean(model.focal_loss_softmax(labels=labels_pl, logits=pred, gamma=PARAMS["focusing_param"], alpha=smpws_pl))	# focal loss weighted by class
			else:
				assert PARAMS["loss_func"] == "ce"
				loss = model.get_loss(pred, labels_pl, smpws_pl, end_points)		# This is the softmax cross entropy weighted by class 
				tf.summary.scalar("loss", loss)

			print("--- Get training operator")
			# Get training operator
			learning_rate = get_learning_rate(batch)
			tf.summary.scalar("learning_rate", learning_rate)
			if PARAMS["optimizer"] == "momentum":
					optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=PARAMS["momentum"])
			else:
					assert PARAMS["optimizer"] == "adam"
					optimizer = tf.train.AdamOptimizer(learning_rate)
					train_op = optimizer.minimize(loss, global_step=batch)

			# Add ops to save and restore all the variables.
			saver = tf.train.Saver()
												
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)
		saver.restore(sess, FLAGS.ckpt)
		print(f'Model restored from {FLAGS.ckpt}')
								
		all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
		print(sess.run(all_trainable_vars))

		sys.exit() 


if __name__ == "__main__":
		count_num_params()
