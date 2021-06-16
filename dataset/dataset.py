import os
import numpy as np
import util.provider as provider
from util.point_cloud_util import load_labels


class FileData:
		def __init__(
				self, file_path_without_ext, has_label, num_additional_inputs, box_size_x, box_size_y
		):
				"""
				Loads file data
				"""
				self.file_path_without_ext = file_path_without_ext
				self.box_size_x = box_size_x
				self.box_size_y = box_size_y

				# Load points
				self.points = np.loadtxt(file_path_without_ext + '.txt', delimiter=',', usecols=(0,1,2))

				# Load label. In pure test set, fill with zeros.
				if has_label:
						self.labels = load_labels(file_path_without_ext + ".labels")
				else:
						self.labels = np.zeros(len(self.points)).astype(bool)

				# Load additional inputs. If not use, fill with zeros.
				if num_additional_inputs > 0:
						self.colors = np.loadtxt(file_path_without_ext + '.txt', delimiter=',', usecols=(np.arange(3,3+num_additional_inputs)))
				else:
						self.colors = np.zeros_like(self.points)

				# Sort according to x to speed up computation of boxes and z-boxes
				sort_idx = np.argsort(self.points[:, 0])
				self.points = self.points[sort_idx]
				self.labels = self.labels[sort_idx]
				self.colors = self.colors[sort_idx]
				
				# Remove points w/ specificed labels. This is the fix for a specific project that has unknow class code.
				remove_idx = np.where((self.labels==130) | (self.labels==12) | (self.labels==103))
				self.points = np.delete(self.points, remove_idx[0], axis=0)
				self.labels = np.delete(self.labels, remove_idx[0], axis=0)
				self.colors = np.delete(self.colors, remove_idx[0], axis=0)
				
		def _get_fix_sized_sample_mask(self, points, num_points_per_sample):
				"""
				Get down-sample or up-sample mask to sample points to num_points_per_sample
				"""
				# TODO: change this to numpy's build-in functions
				# Shuffling or up-sampling if needed
				if len(points) - num_points_per_sample > 0:
						true_array = np.ones(num_points_per_sample, dtype=bool)
						false_array = np.zeros(len(points) - num_points_per_sample, dtype=bool)
						sample_mask = np.concatenate((true_array, false_array), axis=0)
						np.random.shuffle(sample_mask)
				else:
						# Not enough points, recopy the data until there are enough points
						sample_mask = np.arange(len(points))
						while len(sample_mask) < num_points_per_sample:
								sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
						sample_mask = sample_mask[:num_points_per_sample]
				return sample_mask

		def _center_box(self, points):
				# Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
				# E.g. if box_size_x == box_size_y == 10, then the new mins are (-5, -5, 0)
				box_min = np.min(points, axis=0)
				shift = np.array(
						[
								box_min[0] + self.box_size_x / 2,
								box_min[1] + self.box_size_y / 2,
								box_min[2],
						]
				)
				points_centered = points - shift
				return points_centered

		def _extract_z_box(self, center_point):
				"""
				Crop along z axis (vertical) from the center_point.

				Args:
						center_point: only x and y coordinates will be used
						points: points (n * 3)
						scene_idx: scene index to get the min and max of the whole scene
				"""
				# TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
				scene_z_size = np.max(self.points, axis=0)[2] - np.min(self.points, axis=0)[2]
				box_min = center_point - [
						self.box_size_x / 2,
						self.box_size_y / 2,
						scene_z_size,
				]
				box_max = center_point + [
						self.box_size_x / 2,
						self.box_size_y / 2,
						scene_z_size,
				]

				i_min = np.searchsorted(self.points[:, 0], box_min[0])
				i_max = np.searchsorted(self.points[:, 0], box_max[0])
				mask = (
						np.sum(
								(self.points[i_min:i_max, :] >= box_min)
								* (self.points[i_min:i_max, :] <= box_max),
								axis=1,
						)
						== 3
				)
				mask = np.hstack(
						(
								np.zeros(i_min, dtype=bool),
								mask,
								np.zeros(len(self.points) - i_max, dtype=bool),
						)
				)

				# mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
				assert np.sum(mask) != 0
				return mask

		def sample(self, num_points_per_sample):
				points = self.points

				# Pick a point, and crop a z-box around
				center_point = points[np.random.randint(0, len(points))]
				scene_extract_mask = self._extract_z_box(center_point)
				points = points[scene_extract_mask]
				labels = self.labels[scene_extract_mask]
				colors = self.colors[scene_extract_mask]

				sample_mask = self._get_fix_sized_sample_mask(points, num_points_per_sample)
				points = points[sample_mask]
				labels = labels[sample_mask]
				colors = colors[sample_mask]

				# Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
				# This canonical column is used for both training and inference
				points_centered = self._center_box(points)

				return points_centered, points, labels, colors

		def sample_batch(self, batch_size, num_points_per_sample):
				"""
				TODO: change this to stack instead of extend
				"""
				batch_points_centered = []
				batch_points_raw = []
				batch_labels = []
				batch_colors = []

				for _ in range(batch_size):
						points_centered, points_raw, gt_labels, colors = self.sample(
								num_points_per_sample
						)
						batch_points_centered.append(points_centered)
						batch_points_raw.append(points_raw)
						batch_labels.append(gt_labels)
						batch_colors.append(colors)

				return (
						np.array(batch_points_centered),
						np.array(batch_points_raw),
						np.array(batch_labels),
						np.array(batch_colors),
				)


class Dataset:
		def __init__(
				self, num_points_per_sample, split, num_additional_inputs, box_size_x, box_size_y, path, num_classes, labels_names, train_areas, validation_areas, test_areas
		):
				"""Create a dataset holder
				num_points_per_sample (int): Defaults to 8192. The number of point per sample
				split (str): Defaults to 'train'. The selected part of the data (train, test,
										 reduced...)
				color (bool): Defaults to True. Whether to use colors or not
				box_size_x (int): Defaults to 10. The size of the extracted cube.
				box_size_y (int): Defaults to 10. The size of the extracted cube.
				path (float): Defaults to '{project}_downsampled/'.
				num_classes (int): Number of output classes.
				labels_names (list): Name of output classes.
				train_areas (list): Area folders for training data.
				test_areas (list): Area folders for testing data.
				validation_areas (list): Area folders for validation data.
				file_paths_without_ext (list): Paths to all files in a dataset
				"""
				# Dataset parameters
				self.num_points_per_sample = num_points_per_sample
				#print('num_points: ', self.num_points_per_sample)
				self.split = split
				#print('split: ', self.split)
				self.num_additional_inputs = num_additional_inputs
				#print('num_additional_inputs: ', self.num_additional_inputs)
				self.box_size_x = box_size_x
				#print('box_size_x: ', self.box_size_x)
				self.box_size_y = box_size_y
				#print('box_size_y: ', self.box_size_y)
				self.num_classes = num_classes 
				#print('num_classes: ', self.num_classes)
				self.path = path
				#print('path: ', self.path)
				self.labels_names = labels_names
				#print('labels_names: ', self.labels_names)
				self.train_areas = train_areas
				#print('train_areas: ', self.train_areas)
				self.test_areas = test_areas
				#print('test_areas: ', self.test_areas)
				self.validation_areas = validation_areas
				#print('validation_areas: ', self.validation_areas)

				# Get file_paths
				self.file_paths_without_ext = self.get_file_paths_without_ext()
				# Pre-compute the probability of picking a scene
				self.num_scenes = len(self.file_paths_without_ext)
				self.list_scene_points = []
				print(f"Dataset split: {self.split}, {self.num_scenes} files")


				# Pre-compute the points weights if it is a training set
				if self.split == "train" or self.split == "validation":
						# First, compute the histogram of each labels
						label_weights = np.zeros(self.num_classes)
						labels = []

						for file_path_without_ext in self.file_paths_without_ext:
								 labels = load_labels(file_path_without_ext + ".labels")
								 tmp, _ = np.histogram(labels, range(self.num_classes+1))
								 label_weights += tmp
								 self.list_scene_points.append(len(labels))
						self.total_num_points = sum(label_weights)
						print(f'Point per class in {self.split}: ', label_weights)
						self.scene_probs = [ scene_points / self.total_num_points for scene_points in self.list_scene_points]
						# Normalise probs to avoid error "sum(probs)!=1"
						self.scene_probs /= sum(self.scene_probs)

						# Then, a heuristic gives the weights
						# 1 / log(1.2 + probability of occurrence)
						label_weights = label_weights.astype(np.float32)
						label_weights = label_weights / np.sum(label_weights)
						self.label_weights = 1 / np.log(1.2 + label_weights)
						print(f'label_weights after applying heuristic in {self.split}: ', self.label_weights)
				else:
						self.label_weights = np.zeros(self.num_classes)

		def get_file_paths_without_ext(self):
				if self.split == "train":
						train_file_paths = []
						for area in self.train_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												train_file_paths.append(file_path_without_ext)
						return train_file_paths
				elif self.split == "train_full":
						train_file_paths = []
						for area in self.train_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												train_file_paths.append(file_path_without_ext)
						validation_file_paths = []
						for area in self.validation_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												validation_file_paths.append(file_path_without_ext)
						return train_file_paths + validation_file_paths
				elif self.split == "validation": 
						validation_file_paths = []
						for area in self.validation_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												validation_file_paths.append(file_path_without_ext)
						return validation_file_paths
				elif self.split == "test": 
						test_file_paths = []
						for area in self.test_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												test_file_paths.append(file_path_without_ext)
						return test_file_paths
				elif self.split == "all": 
						train_file_paths = []
						for area in self.train_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												train_file_paths.append(file_path_without_ext)
						validation_file_paths = []
						for area in self.validation_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												validation_file_paths.append(file_path_without_ext)
						test_file_paths = []
						for area in self.test_areas:
								area_path = os.path.join(self.path, area)
								for file in os.listdir(area_path):
										if file.endswith(".txt"):
												file_path_without_ext = os.path.join(area_path, os.path.splitext(file)[0])
												test_file_paths.append(file_path_without_ext)
						return train_file_paths + validation_file_paths + test_file_paths


		def sample_batch_in_all_files(self, batch_size, augment=True):
				batch_data = []
				batch_label = []
				batch_weights = []

				for _ in range(batch_size):
						points, labels, colors, weights = self.sample_in_all_files(is_training=True)
						if self.num_additional_inputs > 0:
								batch_data.append(np.hstack((points, colors)))
						else:
								batch_data.append(points)
						batch_label.append(labels)
						batch_weights.append(weights)

				batch_data = np.array(batch_data)
				batch_label = np.array(batch_label)
				batch_weights = np.array(batch_weights)

				if augment:
						if self.num_additional_inputs > 0:
								batch_data = provider.rotate_feature_point_cloud(batch_data, self.num_additional_inputs)
						else:
								batch_data = provider.rotate_point_cloud(batch_data)

				return batch_data, batch_label, batch_weights

		def sample_in_all_files(self, is_training):
				"""
				Returns points and other info within a z - cropped box.
				"""
				# Pick a scene, scenes with more points are more likely to be chosen
				scene_index = np.random.choice(np.arange(0, self.num_scenes), p=self.scene_probs)
				#print("picked_scene: ", self.file_paths_without_ext[scene_index])

				# Load data and Sample from the selected scene
				points_centered, points_raw, labels, colors = FileData(
								file_path_without_ext=self.file_paths_without_ext[scene_index],
								has_label=self.split != "test",
								num_additional_inputs=self.num_additional_inputs,
								box_size_x=self.box_size_x,
								box_size_y=self.box_size_y
						).sample(num_points_per_sample=self.num_points_per_sample)
				#print("labels: ", labels)
				#print("set_labels: ", set(labels))

				if is_training:
						weights = self.label_weights[labels] 
						return points_centered, labels, colors, weights
				else:
						return scene_index, points_centered, points_raw, labels, colors

		def get_num_batches(self, batch_size):
				return int( self.total_num_points / (batch_size * self.num_points_per_sample))


if __name__ == "__main__":
		import argparse
		import json
		import multiprocessing as mp
		
		# Two global arg collections
		parser = argparse.ArgumentParser()
		parser.add_argument("--train_set", default="train", help="train, train_full, validation, test, all")
		parser.add_argument("--config_file", default="test.json", help="config file path")
		
		FLAGS = parser.parse_args()
		PARAMS = json.loads(open(FLAGS.config_file).read())

		pool = mp.Pool(processes=2)
		datasets = pool.starmap(Dataset, [(PARAMS["num_point"], split, PARAMS["num_additional_inputs"], PARAMS["box_size_x"], PARAMS["box_size_y"],PARAMS["data_path"], PARAMS["num_classes"], PARAMS["labels_names"], PARAMS["train_areas"], PARAMS["validation_areas"], PARAMS["test_areas"]) for split in ["train", "validation"]])
		TRAIN_DATASET = datasets[0]
		VALIDATION_DATASET = datasets[1]		
		
		#TRAIN_DATASET = Dataset(
		#num_points_per_sample=PARAMS["num_point"],
		#split="train",
		#box_size_x=PARAMS["box_size_x"],
		#box_size_y=PARAMS["box_size_y"],
		#num_additional_inputs=PARAMS["num_additional_inputs"],
		#path=PARAMS["data_path"],
		#num_classes=PARAMS["num_classes"],
		#labels_names=PARAMS["labels_names"],
		#train_areas=PARAMS["train_areas"],
		#validation_areas=PARAMS["validation_areas"],
		#test_areas=PARAMS["test_areas"]
		#)
		
		
		#NUM_CLASSES = TRAIN_DATASET.num_classes
		
		batch = TRAIN_DATASET.sample_batch_in_all_files(
						1, augment=True
				)
