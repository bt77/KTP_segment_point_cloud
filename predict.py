import argparse
import os, glob
import json
import numpy as np
import tensorflow as tf
import open3d
from scipy.special import softmax

import model
from dataset.dataset import Dataset, FileData
from util.metric import ConfusionMatrix
from tf_ops.tf_interpolate import interpolate_label_with_color


class Predictor:
        def __init__(self, checkpoint_path, num_classes, hyper_params):
                # Get ops from graph
                with tf.device("/gpu:0"):
                        # Placeholder
                        pl_points, _, _ = model.get_placeholders(
                                hyper_params["num_point"], hyperparams=hyper_params
                        )
                        pl_is_training = tf.placeholder(tf.bool, shape=())

                        # Prediction
                        pred, _ = model.get_model(
                                pl_points, pl_is_training, num_classes, hyperparams=hyper_params
                        )

                        # Saver
                        saver = tf.train.Saver()

                        # Graph for interpolating labels
                        # Assuming batch_size == 1 for simplicity
                        pl_sparse_points = tf.placeholder(tf.float32, (None, 3))
                        pl_sparse_labels = tf.placeholder(tf.int32, (None,))
                        pl_dense_points = tf.placeholder(tf.float32, (None, 3))
                        pl_knn = tf.placeholder(tf.int32, ())
                        dense_labels, dense_colors = interpolate_label_with_color(
                                pl_sparse_points, pl_sparse_labels, pl_dense_points, pl_knn
                        )

                self.ops = {
                        "pl_points": pl_points,
                        "pl_is_training": pl_is_training,
                        "pred": pred,
                        "pl_sparse_points": pl_sparse_points,
                        "pl_sparse_labels": pl_sparse_labels,
                        "pl_dense_points": pl_dense_points,
                        "pl_knn": pl_knn,
                        "dense_labels": dense_labels,
                        "dense_colors": dense_colors,
                }

                # Restore checkpoint to session
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, checkpoint_path)
                print("Model restored: ", checkpoint_path)

        def predict(self, batch_data, run_metadata=None, run_options=None):
                """
                Args:
                        batch_data: batch_size * num_point * 6(3)

                Returns:
                        pred_labels: batch_size * num_point * 1
                """
                is_training = False
                feed_dict = {
                        self.ops["pl_points"]: batch_data,
                        self.ops["pl_is_training"]: is_training,
                }
                if run_metadata is None:
                        run_metadata = tf.RunMetadata()
                if run_options is None:
                        run_options = tf.RunOptions()

                pred_val = self.sess.run(
                        [self.ops["pred"]],
                        options=run_options,
                        run_metadata=run_metadata,
                        feed_dict=feed_dict,
                )
                pred_val = pred_val[0]  # batch_size * num_point * num_class, unscaled log probablity
                return pred_val


def ensemble_predict(predictors, points_centered_with_colors):
	# No ensemble if singular model is used
        if len(predictors) == 1:
                summed = next(iter(predictors.values())).predict(points_centered_with_colors)
        else:
		# Predict using all models
                predictions = {}
                for model in predictors:
                	predictions[model] = softmax(predictors[model].predict(points_centered_with_colors))
                #print('probs: ', predictions)
                # Ensemble_weights * Predictions
                ensemble_weights = hyper_params["ensemble_weights"]
                pred_val = []
                for model in predictors:
                	pred_val.append(predictions[model] * ensemble_weights[model])
                # sum across ensemble members
                pred_val = np.array(pred_val)
                #print('pred_val: ', pred_val)
                summed = np.sum(pred_val, axis=0)
                #print('summed: ', summed)
        
	# argmax across classes
        pd_labels = np.argmax(summed, axis=2)
        #print('pd_labels', pd_labels)
        return pd_labels	

def prepare_models():
        print("-------------------------- LOADING MODELS -------------------------")
        predictors = {}
        for ckpt in glob.glob(os.path.join(flags.ckpt_dir, '*.ckpt.meta')):
        	model_name = os.path.basename(ckpt).split('.')[0]
        	with tf.name_scope(model_name):
        		predictors[model_name] = Predictor(
                		checkpoint_path=ckpt[:-5],
                		num_classes=dataset.num_classes,
                		hyper_params=hyper_params,
        		)
       	print(len(predictors), 'models')
       	return predictors
       	
                


if __name__ == "__main__":
        np.random.seed(0)

        # Parser
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "--num_samples",
                type=int,
                default=1,
                help="# samples, each contains num_point points_centered",
        )
        parser.add_argument("--ckpt_dir", default="models", help="Path to checkpoint files")
        parser.add_argument("--set", default="test", help="train, validation, test")
        parser.add_argument("--config_file", required=True, help="Path to model config file")
        flags = parser.parse_args()
        hyper_params = json.loads(open(flags.config_file).read())

        
        # Dataset
        dataset = Dataset(
                num_points_per_sample=hyper_params["num_point"],
                split=flags.set,
                box_size_x=hyper_params["box_size_x"],
                box_size_y=hyper_params["box_size_y"],
                num_additional_inputs=hyper_params["num_additional_inputs"],
                path=hyper_params["data_path"],
                num_classes=hyper_params["num_classes"],
                labels_names=hyper_params["labels_names"],
                train_areas=hyper_params["train_areas"],
                validation_areas=hyper_params["validation_areas"],
                test_areas=hyper_params["test_areas"]
        )

        batch_size = 64
        # Models
        predictors = prepare_models()
        print("-------------------------- PREDICTING -------------------------")

        cm = ConfusionMatrix(hyper_params["num_classes"])
        for file_path_without_ext in dataset.file_paths_without_ext:
                file_data = FileData(file_path_without_ext=file_path_without_ext, has_label=dataset.split!="test", num_additional_inputs=dataset.num_additional_inputs, box_size_x=dataset.box_size_x, box_size_y=dataset.box_size_y)

                # Predict for num_samples times
                points_collector = []
                pd_labels_collector = []

                # If flags.num_samples < batch_size, will predict one batch
                for batch_index in range(int(np.ceil(flags.num_samples / batch_size))):
                        current_batch_size = min(
                                batch_size, flags.num_samples - batch_index * batch_size
                        )

                        # Get data
                        points_centered, points, gt_labels, colors = file_data.sample_batch(
                                batch_size=current_batch_size,
                                num_points_per_sample=hyper_params["num_point"],
                        )

                        # (bs, 8192, 3) concat (bs, 8192, 3) -> (bs, 8192, 6)
                        if hyper_params["num_additional_inputs"] > 0:
                                points_centered_with_colors = np.concatenate(
                                        (points_centered, colors), axis=-1
                                )
                        else:
                                points_centered_with_colors = points_centered
			
			# Predict
                        pd_labels = ensemble_predict(predictors, points_centered_with_colors)

                        # Save to collector for file output
                        points_collector.extend(points)
                        pd_labels_collector.extend(pd_labels)
                        # Increment confusion matrix
                        cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())

                # Save sparse point cloud and predicted labels
                input_path = file_data.file_path_without_ext
                file_prefix = os.path.basename(input_path)
                # Create output dir
                output_dir = os.path.join("result", "sparse", hyper_params["output_dir"], flags.set)
                os.makedirs(output_dir, exist_ok=True)

                sparse_points = np.array(points_collector).reshape((-1, 3))
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(sparse_points)
                pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
                open3d.io.write_point_cloud(pcd_path, pcd)

                sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
                pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
                np.savetxt(pd_labels_path, sparse_labels, fmt="%d")
                print("Exported sparse labels to {}".format(pd_labels_path))

        cm.print_metrics()
