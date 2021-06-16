import os
import sys
import json
import datetime
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import argparse
import time
from datetime import datetime

import util.metric as metric
import model
from dataset.dataset import Dataset

# Two global arg collections
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", default="train", help="train, train_full")

parser.add_argument("--config_file", default="test.json", help="config file path")
parser.add_argument("--ckpt", help="path to saved model")
parser.add_argument("--best_iou", help="best iou obtained from the checkpoint")

FLAGS = parser.parse_args()
PARAMS = json.loads(open(FLAGS.config_file).read())
os.makedirs(PARAMS["logdir"], exist_ok=True)

# Import dataset
NUM_CLASSES = PARAMS["num_classes"]

pool = mp.Pool(processes=2)
datasets = pool.starmap(Dataset, [(PARAMS["num_point"], split, PARAMS["num_additional_inputs"], PARAMS["box_size_x"], PARAMS["box_size_y"],PARAMS["data_path"], PARAMS["num_classes"], PARAMS["labels_names"], PARAMS["train_areas"], PARAMS["validation_areas"], PARAMS["test_areas"]) for split in ["train", "validation"]])
TRAIN_DATASET = datasets[0]
VALIDATION_DATASET = datasets[1]


# Start logging
LOG_FOUT = open(os.path.join(PARAMS["logdir"], "log_train.txt"), "w")

if FLAGS.ckpt:
        EPOCH_CNT = PARAMS["start_epoch"] + 4
else:
        EPOCH_CNT = 0

def log_string(out_str):
        LOG_FOUT.write(out_str + "\n")
        LOG_FOUT.flush()
        print(out_str)


def update_progress(progress):
        """
        Displays or updates a console progress bar
        Args:
                progress: A float between 0 and 1. Any int will be converted to a float.
                                  A value under 0 represents a 'halt'.
                                  A value at 1 or bigger represents 100%
        """
        barLength = 10  # Modify this to change the length of the progress bar
        if isinstance(progress, int):
                progress = round(float(progress), 2)
        if not isinstance(progress, float):
                progress = 0
        if progress < 0:
                progress = 0
        if progress >= 1:
                progress = 1
        block = int(round(barLength * progress))
        text = "\rProgress: [{}] {}%".format(
                "#" * block + "-" * (barLength - block), progress * 100
        )
        sys.stdout.write(text)
        sys.stdout.flush()


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


def get_batch(split):
        np.random.seed()
        if split == "train":
                return TRAIN_DATASET.sample_batch_in_all_files(
                        PARAMS["batch_size"], augment=True
                )
        else:
                return VALIDATION_DATASET.sample_batch_in_all_files(
                        PARAMS["batch_size"], augment=False
                )


def fill_queues(
        stack_train, stack_validation, num_train_batches, num_validation_batches
):
        """
        Args:
                stack_train: mp.Queue to be filled asynchronously
                stack_validation: mp.Queue to be filled asynchronously
                num_train_batches: max number of training batches to queue
                num_validation_batches: max number of validationation batches to queue
        """
        pool = mp.Pool(processes=mp.cpu_count())
        launched_train = 0
        launched_validation = 0
        results_train = []  # Temp buffer before filling the stack_train
        results_validation = []  # Temp buffer before filling the stack_validation
        # Launch as much as n
        while True:
                if stack_train.qsize() + launched_train < num_train_batches:
                        results_train.append(pool.apply_async(get_batch, args=("train",)))
                        launched_train += 1
                elif stack_validation.qsize() + launched_validation < num_validation_batches:
                        results_validation.append(pool.apply_async(get_batch, args=("validation",)))
                        launched_validation += 1
                for p in results_train:
                        if p.ready():
                                stack_train.put(p.get())
                                results_train.remove(p)
                                launched_train -= 1
                for p in results_validation:
                        if p.ready():
                                stack_validation.put(p.get())
                                results_validation.remove(p)
                                launched_validation -= 1
                # Stability
                time.sleep(0.01)


def init_stacking():
        """
        Returns:
                stacker: mp.Process object
                stack_validation: mp.Queue, use stack_validation.get() to read a batch
                stack_train: mp.Queue, use stack_train.get() to read a batch
        """
        with tf.device("/cpu:0"):
                # Queues that contain several batches in advance
                num_cpu = mp.cpu_count()
                stack_train = mp.Queue(num_cpu)
                stack_validation = mp.Queue(num_cpu)
                stacker = mp.Process(
                        target=fill_queues,
                        args=(
                                stack_train,
                                stack_validation, num_cpu, num_cpu
                        ),
                )
                stacker.start()
                return stacker, stack_validation, stack_train


def train_one_epoch(sess, ops, train_writer, stack):
        """Train one epoch

        Args:
                sess (tf.Session): the session to evaluate Tensors and ops
                ops (dict of tf.Operation): contain multiple operation mapped with with strings
                train_writer (tf.FileSaver): enable to log the training with TensorBoard
                compute_class_iou (bool): it takes time to compute the iou per class, so you can
                                                                  disable it here
        """

        is_training = True

        num_batches = TRAIN_DATASET.get_num_batches(PARAMS["batch_size"])

        log_string(str(datetime.now()))
        update_progress(0)
        # Reset metrics
        loss_sum = 0
        confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

        # Train over num_batches batches
        for batch_idx in range(num_batches):
                # Refill more batches if empty
                progress = float(batch_idx) / float(num_batches)
                update_progress(round(progress, 2))
                batch_data, batch_label, batch_weights = stack.get()

                # Get predicted labels
                feed_dict = {
                        ops["pointclouds_pl"]: batch_data,
                        ops["labels_pl"]: batch_label,
                        ops["smpws_pl"]: batch_weights,
                        ops["is_training_pl"]: is_training,
                }

                summary, step, _, loss_val, pred_val = sess.run(
                        [
                                ops["merged"],
                                ops["step"],
                                ops["train_op"],
                                ops["loss"],
                                ops["pred"],
                        #       ops["update_iou"]
                        ],
                        feed_dict=feed_dict,
                )
                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)

                # Update metrics
                for i in range(len(pred_val)):
                        for j in range(len(pred_val[i])):
                                confusion_matrix.increment(batch_label[i][j], pred_val[i][j])
                loss_sum += loss_val
        
        # Display metrics
        update_progress(1)
        loss = loss_sum / float(num_batches)
        # Log loss to display in Tensorboard
        epoch_loss = tf.Summary(value=[tf.Summary.Value(tag="Epoch Loss", simple_value=loss),])
        train_writer.add_summary(epoch_loss, step)
        log_string("mean loss: %f" % loss)
        acc = confusion_matrix.get_accuracy()
        log_string("Overall accuracy : %f" % (acc))
        # only record valid mIoU
        mIoU = confusion_matrix.get_mean_iou()
        if mIoU != None:
                log_string("Average IoU : %f" % (mIoU))
                # Log mIoU in Tensorboard
                mIoU_epoch = tf.Summary(value=[tf.Summary.Value(tag="Epoch mIoU", simple_value=mIoU),])
                train_writer.add_summary(mIoU_epoch, step)
        iou_per_class = confusion_matrix.get_per_class_ious()
        num_per_class = confusion_matrix.get_num_per_class()
        for i in range(NUM_CLASSES):
                log_string("IoU of %s : %s" % (TRAIN_DATASET.labels_names[i], iou_per_class[i]))
                # Log IoU to Tensorboard
                iou = tf.Summary(value=[tf.Summary.Value(tag=f"Epoch IoU of {TRAIN_DATASET.labels_names[i]}", simple_value=iou_per_class[i]),])
                train_writer.add_summary(iou, step)
                # Log num_per_class in Tensorboard
                num_point = tf.Summary(value=[tf.Summary.Value(tag=f"Epoch #Point of {TRAIN_DATASET.labels_names[i]}", simple_value=num_per_class[i]),])
                train_writer.add_summary(num_point, step)
        # Log accuracy to display in Tensorboard
        accuracy = tf.Summary(value=[tf.Summary.Value(tag="Epoch Accuracy", simple_value=acc),])
        train_writer.add_summary(accuracy, step)
        

        

def eval_one_epoch(sess, ops, validation_writer, stack):
        """Evaluate one epoch

        Args:
                sess (tf.Session): the session to evaluate tensors and operations
                ops (tf.Operation): the dict of operations
                validation_writer (tf.summary.FileWriter): enable to log the evaluation on TensorBoard

        Returns:
                float: the IoU of powerline structures computed on the validationation set
        """

        global EPOCH_CNT

        is_training = False

        num_batches = VALIDATION_DATASET.get_num_batches(PARAMS["batch_size"])

        # Reset metrics
        loss_sum = 0
        confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

        log_string(str(datetime.now()))

        log_string("---- EPOCH %03d EVALUATION ----" % (EPOCH_CNT))

        update_progress(0)

        for batch_idx in range(num_batches):
                progress = float(batch_idx) / float(num_batches)
                update_progress(round(progress, 2))
                batch_data, batch_label, batch_weights = stack.get()

                feed_dict = {
                        ops["pointclouds_pl"]: batch_data,
                        ops["labels_pl"]: batch_label,
                        ops["smpws_pl"]: batch_weights,
                        ops["is_training_pl"]: is_training,
                }
                summary, step, loss_val, pred_val = sess.run(
                        [ops["merged"], ops["step"], ops["loss"], ops["pred"]], feed_dict=feed_dict
                )
                validation_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)  # BxN

                # Update metrics
                for i in range(len(pred_val)):
                        for j in range(len(pred_val[i])):
                                confusion_matrix.increment(batch_label[i][j], pred_val[i][j])
                loss_sum += loss_val

        update_progress(1)
        num_per_class = confusion_matrix.get_num_per_class()
        iou_per_class = confusion_matrix.get_per_class_ious()
        
        # Display metrics
        loss = loss_sum / float(num_batches)
        log_string("mean loss: %f" % loss)
        # Log accuracy to display in Tensorboard
        epoch_loss = tf.Summary(value=[tf.Summary.Value(tag="Epoch Loss", simple_value=loss),])
        validation_writer.add_summary(epoch_loss, step)
        acc = confusion_matrix.get_accuracy()
        log_string("Overall accuracy : %f" % (acc))

        # only record valid mIoU
        mIoU = confusion_matrix.get_mean_iou()
        if mIoU != None:
                log_string("Average IoU : %f" % (mIoU))
                # Log mIoU in Tensorboard
                mIoU_epoch = tf.Summary(value=[tf.Summary.Value(tag="Epoch mIoU", simple_value=mIoU),])
                validation_writer.add_summary(mIoU_epoch, step)
        for i in range(NUM_CLASSES):
                log_string(
                        "IoU of %s : %s" % (VALIDATION_DATASET.labels_names[i], iou_per_class[i])
                )
                # Log IoU in Tensorboard
                iou = tf.Summary(value=[tf.Summary.Value(tag=f"Epoch IoU of {TRAIN_DATASET.labels_names[i]}", simple_value=iou_per_class[i]),])
                validation_writer.add_summary(iou, step)
                # Log num_per_class in Tensorboard
                num_point = tf.Summary(value=[tf.Summary.Value(tag=f"Epoch #Point of {TRAIN_DATASET.labels_names[i]}", simple_value=num_per_class[i]),])
                validation_writer.add_summary(num_point, step)


        EPOCH_CNT += 5

        
        # Log accuracy to display in Tensorboard
        accuracy = tf.Summary(value=[tf.Summary.Value(tag="Epoch Accuracy", simple_value=acc),])
        validation_writer.add_summary(accuracy, step)
        


        return iou_per_class[1] # use index of class that is of interest


def train():
        """Train the model on a single GPU
        """
        with tf.Graph().as_default():
                stacker, stack_validation, stack_train = init_stacking()

                with tf.device("/gpu:" + str(PARAMS["gpu"])):
                        pointclouds_pl, labels_pl, smpws_pl = model.get_placeholders(
                                PARAMS["num_point"], hyperparams=PARAMS
                        )
                        is_training_pl = tf.placeholder(tf.bool, shape=())

                        # Note the global_step=batch parameter to minimize.
                        # That tells the optimizer to helpfully increment the 'batch' parameter for
                        # you every time it trains.
                        batch = tf.Variable(0)
                        bn_decay = get_bn_decay(batch)
                        tf.summary.scalar("bn_decay", bn_decay)

                        print("--- Get model and loss")
                        # Get model and loss
                        pred, end_points = model.get_model(
                                pointclouds_pl,
                                is_training_pl,
                                NUM_CLASSES,
                                hyperparams=PARAMS,
                                bn_decay=bn_decay,
                        )
                        if PARAMS["loss_func"] == "focal":
                                if PARAMS["balancing_factor"]:
                                    print("Using default focal loss")
                                    loss = tf.reduce_mean(model.focal_loss_softmax(labels=labels_pl, logits=pred, gamma=PARAMS["focusing_param"], alpha=PARAMS["balancing_factor"])) # focal loss weighted by specified balancing factor
                                else:
                                    print("Using focal loss weighted by class")
                                    loss = tf.reduce_mean(model.focal_loss_softmax(labels=labels_pl, logits=pred, gamma=PARAMS["focusing_param"], alpha=smpws_pl))  # focal loss weighted by class
                        else:
                                assert PARAMS["loss_func"] == "ce"
                                loss = model.get_loss(pred, labels_pl, smpws_pl, end_points)    # This is the softmax cross entropy weighted by class 
                        tf.summary.scalar("loss", loss)

                        print("--- Get training operator")
                        # Get training operator
                        learning_rate = get_learning_rate(batch)
                        tf.summary.scalar("learning_rate", learning_rate)
                        if PARAMS["optimizer"] == "momentum":
                                optimizer = tf.train.MomentumOptimizer(
                                        learning_rate, momentum=PARAMS["momentum"]
                                )
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
                if FLAGS.ckpt:
                        saver.restore(sess, FLAGS.ckpt)
                        print(f'Model restored from {FLAGS.ckpt} with IoU of {FLAGS.best_iou}')

                # Add summary writers
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                        os.path.join(PARAMS["logdir"], "train"), sess.graph
                )
                validation_writer = tf.summary.FileWriter(
                        os.path.join(PARAMS["logdir"], "validation"), sess.graph
                )
                
                if not FLAGS.ckpt:
                        # Init variables
                        sess.run(tf.global_variables_initializer())
                #sess.run(tf.local_variables_initializer())  # important for mIoU

                ops = {
                        "pointclouds_pl": pointclouds_pl,
                        "labels_pl": labels_pl,
                        "smpws_pl": smpws_pl,
                        "is_training_pl": is_training_pl,
                        "pred": pred,
                        "loss": loss,
                        "train_op": train_op,
                        "merged": merged,
                        "step": batch,
                        "end_points": end_points
                }

                # Train for hyper_params["max_epoch"] epochs
                if FLAGS.ckpt:
                        best_iou = float(FLAGS.best_iou)
                        start_epoch = PARAMS["start_epoch"]
                        iou_pl = None
                else:
                        best_iou = 0
                        start_epoch = 0
               
                for epoch in range(start_epoch, PARAMS["max_epoch"] + 1):
                        print("in epoch", epoch)
                        print("max_epoch", PARAMS["max_epoch"])

                        log_string("**** EPOCH %03d ****" % (epoch))
                        sys.stdout.flush()

                        # Train one epoch
                        train_one_epoch(sess, ops, train_writer, stack_train)
                        save_path = saver.save(sess, os.path.join(PARAMS["logdir"], "model.ckpt"))
                        log_string("Model saved in file: %s" % save_path)
                        print("Model saved in file: %s" % save_path)

                        # Evaluate, save, and compute the iou for powerline structure class
                        if epoch % 5 == 0:
                                iou_pl = eval_one_epoch(sess, ops, validation_writer, stack_validation)

                        #print('iou_pl', iou_pl)
                        if iou_pl != None and iou_pl > best_iou:
                                best_iou = iou_pl
                                save_path = saver.save(
                                        sess,
                                        os.path.join(
                                                PARAMS["logdir"], "best_model_epoch_%03d.ckpt" % (epoch)
                                        ),
                                )
                                log_string("Model saved in file: %s" % save_path)
                                print("Model saved in file: %s" % save_path)

                        # Save the variables to disk.
                        #if epoch % 10 == 0:
                        #        save_path = saver.save(
                        #                sess, os.path.join(PARAMS["logdir"], "model.ckpt")
                        #        )
                        #        log_string("Model saved in file: %s" % save_path)
                        #        print("Model saved in file: %s" % save_path)
                        #print('END of one epoch')

                # Kill the process, close the file and exit
                stacker.terminate()
                LOG_FOUT.close()
                sys.exit()


if __name__ == "__main__":
        train()
