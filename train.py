import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.loss import compute_myrnenko_loss
from utils.optimizer import ScheduledAdam
from utils.constants import *
from model.volumetric_cnn import VolumetricCNN


def prepare_dataset(path, batch_size):
    """Returns a BatchDataset object containing loaded data."""
    def parse_example(example_proto):
        """Mapping function to parse a single example."""
        return tf.io.parse_single_example(example_proto, example_desc)

    def get_dataset_len(tf_dataset):
        """Returns length of dataset until tf.data.experimental.cardinality is fixed."""
        # return tf.data.experimental.cardinality(tf_dataset)
        return sum(1 for _ in tf_dataset)

    example_desc = {
        'X': tf.io.FixedLenFeature([H * W * D * IN_CH], tf.float32),
        'y': tf.io.FixedLenFeature([H * W * D * 1], tf.float32)
    }

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_example).shuffle(570).batch(batch_size)

    return dataset, get_dataset_len(dataset)


def main(args):
    # Load data.
    train_data, n_train = prepare_dataset(args.train_loc, args.batch_size)
    val_data, n_val = prepare_dataset(args.val_loc, args.batch_size)

    # Set up.
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        dropout=args.dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))
    optimizer = ScheduledAdam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))

    if args.log_file:
        with open(args.log, 'w') as f:
            f.write('epoch,lr,train_loss,val_loss\n')

    # Train.
    for epoch in range(args.n_epochs):
        print('Epoch {}.'.format(epoch))

        # Training epoch.
        for step, batch in tqdm(enumerate(train_data), total=n_train):
            # Read data in from serialized string.
            x_batch = batch['X']
            y_batch = batch['y']
            if args.data_format == 'channels_last':
                x_batch = np.reshape(x_batch, CHANNELS_LAST_X_SHAPE)
                y_batch = np.reshape(y_batch, CHANNELS_LAST_Y_SHAPE)
            elif args.data_format == 'channels_first':
                x_batch = np.reshape(x_batch, CHANNELS_FIRST_X_SHAPE)
                y_batch = np.reshape(y_batch, CHANNELS_FIRST_Y_SHAPE)

            optimizer.update_lr(epoch_num=epoch)

            with tf.device(args.device):
                with tf.GradientTape() as tape:
                    # Forward and loss.
                    y_pred, y_vae, z_mean, z_var = model(x_batch)
                    loss = compute_myrnenko_loss(
                                x_batch, y_batch, y_pred, y_vae, z_mean, z_var, data_format=args.data_format)
                    loss += sum(model.losses)

                # Gradients and backward.
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.update_state(loss)

        avg_train_loss = train_loss.result() / n_train
        train_loss.reset_states()
        print('Training loss: {}.'.format(avg_train_loss))

        # Validation epoch.
        for step, batch in tqdm(enumerate(val_data), total=n_val):
            # Read data in from serialized string.
            x_batch = batch['X']
            y_batch = batch['y']
            if args.data_format == 'channels_last':
                x_batch = np.reshape(x_batch, CHANNELS_LAST_X_SHAPE)
                y_batch = np.reshape(y_batch, CHANNELS_LAST_Y_SHAPE)
            elif args.data_format == 'channels_first':
                x_batch = np.reshape(x_batch, CHANNELS_FIRST_X_SHAPE)
                y_batch = np.reshape(y_batch, CHANNELS_FIRST_Y_SHAPE)

            with tf.device(args.device):
                # Forward and loss.
                y_pred, y_vae, z_mean, z_var = model(x_batch)
                loss = compute_myrnenko_loss(
                                x_batch, y_batch, y_pred, y_vae, z_mean, z_var, data_format=args.data_format)
                loss += sum(model.losses)

                val_loss.update_state(loss)

        avg_val_loss = val_loss.result() / n_val
        val_loss.reset_states()
        print('Validation loss: {}.'.format(avg_val_loss))

        # Write logs.
        if args.log_file:
            with open(args.log, 'w') as f:
                f.write('{},{},{},{}\n'.format(
                        epoch, optimizer.learning_rate, avg_train_loss, avg_val_loss))


if __name__ == '__main__':
    args = train_parser()
    main(args)
