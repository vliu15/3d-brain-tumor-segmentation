import tensorflow as tf
import numpy as np

from utils.arg_parser import train_parser
from utils.losses import FocalLoss
from utils.metrics import Accuracy, Precision, Recall, DiceCoefficient
from utils.optimizer import Scheduler
from utils.constants import *
from utils.utils import prepare_dataset
from model.volumetric_cnn import EncDecCNN


def train(args):
    # Load data.
    train_data, n_train = prepare_dataset(
                            args.train_loc, args.batch_size, buffer_size=500, data_format=args.data_format)
    val_data, n_val = prepare_dataset(
                            args.val_loc, args.batch_size, buffer_size=50, data_format=args.data_format)
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))
    train_steps_per_epoch = np.ceil(float(n_train) / args.batch_size).astype(np.int32)
    val_steps_per_epoch = np.ceil(float(n_val) / args.batch_size).astype(np.int32)

    with tf.device(args.device):
        if args.load_file:
            model = tf.keras.models.load_model(args.load_file)
        else:
            model = EncDecCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))

        model.compile(optimizer=tf.keras.optimizers.Adam(
                                    learning_rate=args.lr,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-7,
                                    amsgrad=False),
                      loss=FocalLoss(
                                    gamma=2,
                                    alpha=0.25,
                                    data_format=args.data_format),
                      metrics=[Accuracy(data_format=args.data_format),
                               Precision(data_format=args.data_format),
                               Recall(data_format=args.data_format),
                               DiceCoefficient(data_format=args.data_format)])

        history = model.fit(train_data,
                            epochs=args.n_epochs,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                            Scheduler(args.n_epochs, args.lr)),
                                       tf.keras.callbacks.ModelCheckpoint(
                                            args.save_file,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=False),
                                       tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            min_delta=1e-2,
                                            patience=args.patience),
                                       tf.keras.callbacks.CSVLogger(
                                            args.log_file)],
                            validation_data=val_data,
                            shuffle=True,
                            steps_per_epoch=train_steps_per_epoch,
                            validation_steps=val_steps_per_epoch,
                            validation_freq=1)

        return history


if __name__ == '__main__':
    args = train_parser()
    history = train(args)
