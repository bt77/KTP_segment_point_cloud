from __future__ import print_function
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix as skl_get_confusion_matrix


class ConfusionMatrix:
	def __init__(self, num_classes):
		"""
		label must be {0, 1, 2, ..., num_classes - 1}
		"""
		self.num_classes = num_classes
		self.confusion_matrix = np.zeros(
			(self.num_classes, self.num_classes), dtype=np.int64
		)
		self.valid_labels = set(range(self.num_classes))

	def increment(self, gt_label, pd_label):
		if gt_label not in self.valid_labels:
			raise ValueError("Invalid value for gt_label")
		if pd_label not in self.valid_labels:
			raise ValueError("Invalid value for pd_label")
		self.confusion_matrix[gt_label][pd_label] += 1

	def increment_from_list(self, gt_labels, pd_labels):
		increment_cm = skl_get_confusion_matrix(
			gt_labels, pd_labels, labels=list(range(self.num_classes))
		)
		np.testing.assert_array_equal(self.confusion_matrix.shape, increment_cm.shape)
		self.confusion_matrix += increment_cm

	def get_per_class_ious(self):
		ious = []
		for c in range(len(self.confusion_matrix)):
			# only when num_piont of c != 0, iou is 
			if sum(self.confusion_matrix[c]) == 0:
				print(f'No points of class {c}, IoU set as None')
				ious.append(None)
			else:
				intersection = self.confusion_matrix[c, c]
				union = (
					np.sum(self.confusion_matrix[c, :])
					+ np.sum(self.confusion_matrix[:, c])
					- intersection
				)
				if union == 0: 
					union = 1
				ious.append(float(intersection) / union)
		return ious

	def get_mean_iou(self):
		# only calculate when all classes have valid mIoU
		per_class_ious = self.get_per_class_ious()
		if None not in per_class_ious:
			return np.sum(per_class_ious) / len(per_class_ious)
		else:
			print('No mIoU as not all classes have valid IoU!')

	def get_accuracy(self):
		return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

	def print_metrics(self, labels=None):
		# 1. Confusion matrix
		print("Confusion matrix:")

		# Fill default labels: ["0", "1", "2", ...]
		if labels == None:
			labels = [str(val) for val in range(self.num_classes)]
		elif len(labels) != self.num_classes:
			raise ValueError("len(labels) != self.num_classes")

		# Formatting helpers
		column_width = max([len(x) for x in labels] + [7])
		empty_cell = " " * column_width

		# Print header
		print("    " + empty_cell, end=" ")
		for label in labels:
			print("%{0}s".format(column_width) % label, end=" ")
		print()

		# Print rows
		for i, label in enumerate(labels):
			print("    %{0}s".format(column_width) % label, end=" ")
			for j in range(len(labels)):
				cell = "%{0}.0f".format(column_width) % self.confusion_matrix[i, j]
				print(cell, end=" ")
			print()

		# 2. IoU per class
		print("IoU per class:")
		pprint(self.get_per_class_ious())

		# 3. Mean IoU
		print("mIoU:")
		print(self.get_mean_iou())

		# 4. Overall accuracy
		print("Overall accuracy")
		print(self.get_accuracy())

	def get_num_per_class(self):
		return [sum(i) for i in self.confusion_matrix]


if __name__ == "__main__":
	# Test data
	# |        | 0 (pd) | 1 (pd) |
	# |--------|--------|--------|
	# | 0 (gt) | 10		| 1		 | 
	# | 1 (gt) | 0		| 0		 | 
	 
	ref_confusion_matrix = np.array(
		[[10, 1], [0, 0]]
	)

	# Build CM
	cm = ConfusionMatrix(num_classes=2)
	for gt in range(2):
		for pd in range(2):
			for _ in range(ref_confusion_matrix[gt, pd]):
				cm.increment(gt, pd)

	# Check confusion matrix
	np.testing.assert_allclose(ref_confusion_matrix, cm.confusion_matrix)
	print(cm.confusion_matrix)
	
	# Check num per class
	print(cm.get_num_per_class())


	# Check IoU
	#ref_per_class_ious = np.array(
	#	[
	#		10.0 / (10.0 + 0 + 1),
	#		None
	#	]
	#)
	#np.testing.assert_allclose(cm.get_per_class_ious(), ref_per_class_ious)
	print(cm.get_per_class_ious())

	#ref_mean_iou = np.mean(ref_per_class_ious)
	#assert cm.get_mean_iou() == ref_mean_iou
	print(cm.get_mean_iou())

	# Check accuracy
	ref_accuracy = float(10 + 0 ) / (10 + 1 + 0 + 0)
	assert cm.get_accuracy() == ref_accuracy
	print(cm.get_accuracy())
