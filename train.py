import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.loss import myrnenko_loss
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
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))

    # Initialize model, optimizer, and loss trackers.
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        dropout=args.dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))
    optimizer = ScheduledAdam(learning_rate=args.lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    # Load model weights if specified.
    if args.load_file:
        model.load_weights(args.load_file)
    
    # Set up logging.
    if args.log_file:
        with open(args.log, 'w') as f:
            f.write('epoch,lr,train_loss,val_loss\n')

    best_val_loss = float('inf')
    patience = 0

    # Train.
    for epoch in range(args.n_epochs):
        print('Epoch {}.'.format(epoch))

        # Schedule learning rate.
        optimizer.update_lr(epoch_num=epoch)

        # Training epoch.
        for step, batch in tqdm(enumerate(train_data, 1), total=n_train):
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
                with tf.GradientTape() as tape:
                    # Forward and loss.
                    y_pred, y_vae, z_mean, z_logvar = model(x_batch, training=True)
                    loss = myrnenko_loss(
                                         x_batch, y_batch, y_pred, y_vae, z_mean,
                                         z_logvar, data_format=args.data_format)
                    loss += sum(model.losses)

                # Gradients and backward.
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.update_state(loss)

            # Output loss to console.
            if step % args.log_steps == 0:
                print('Step {}. Cumulative average loss: {}'
                    .format(step, train_loss.result() / step))

        avg_train_loss = train_loss.result() / n_train
        train_loss.reset_states()
        print('Average training loss: {}.'.format(avg_train_loss))

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
                y_pred, y_vae, z_mean, z_logvar = model(x_batch, training=False)
                loss = myrnenko_loss(
                                     x_batch, y_batch, y_pred, y_vae, z_mean,
                                     z_logvar, data_format=args.data_format)
                loss += sum(model.losses)

                val_loss.update_state(loss)

        avg_val_loss = val_loss.result() / n_val
        val_loss.reset_states()
        print('Average validation loss: {}'.format(avg_val_loss))

        # Write logs.
        if args.log_file:
            with open(args.log_file, 'w') as f:
                f.write('{},{},{},{}\n'.format(
                        epoch, optimizer.learning_rate, avg_train_loss, avg_val_loss))

        # Checkpoint and patience.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(args.save_file)
            patience = 0
            print('Saved model weights.')
        elif patience == args.patience:
            print('Validation loss has not improved in {} epochs. Stopped training'
                    .format(args.patience))
            break
        else:
            patience += 1


if __name__ == '__main__':
    args = train_parser()
    main(args)
