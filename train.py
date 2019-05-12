import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.loss import compute_myrnenko_loss
from utils.optimizer import ScheduledAdam
from model.volumetric_cnn import VolumetricCNN


def prepare_dataset(path, batch_size):
    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, example_desc)

    example_desc = {
        'X': tf.io.FixedLenFeature([160 * 192 * 128 * 4], tf.float32),
        'y': tf.io.FixedLenFeature([160 * 192 * 128 * 1], tf.float32)
    }

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_example).batch(batch_size)

    return dataset


def main(args):
    # Load data.
    train_data = prepare_dataset(args.train_loc, args.batch_size)
    val_data = prepare_dataset(args.val_loc, args.batch_size)

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

    # Initial outputs.
    n_train = tf.data.experimental.cardinality(train_data)
    n_val = tf.data.experimental.cardinality(val_data)
    print(f'{n_train} training examples. {n_val} validation examples.')

    if args.log_file:
        with open(args.log, 'w') as f:
            print(f'epoch,train_loss,val_loss\n', file=f)

    # Train.
    for epoch in range(args.n_epochs):
        print(f'Epoch {epoch}.')

        # Training epoch.
        for step, batch in tqdm(enumerate(train_data), total=n_train):
            # Read data in from serialized string.
            x_batch = batch['X']
            y_batch = batch['y']
            if args.data_format == 'channels_last':
                x_batch = np.reshape(x_batch, (-1, 160, 192, 128, 4))
                y_batch = np.reshape(y_batch, (-1, 160, 192, 128, 1))
            elif args.data_format == 'channels_first':
                x_batch = np.reshape(x_batch, (-1, 4, 160, 192, 128))
                y_batch = np.reshape(y_batch, (-1, 1, 160, 192, 128))

            with tf.GradientTape() as tape:
                # Forward and loss.
                y_pred, y_vae, z_mean, z_var = model(x_batch)
                loss = compute_myrnenko_loss(x_batch, y_batch, y_pred, y_vae, z_mean, z_var)
                loss += sum(model.losses)

            # Gradients and backward.
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.update_lr(epoch_num=epoch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)

        avg_train_loss = train_loss.result() / n_train
        train_loss.reset_states()
        print(f'Training loss: {avg_train_loss}.')

        # Validation epoch.
        for step, (x_batch, y_batch) in tqdm(enumerate(val_data), total=n_val):
            # Forward and loss.
            y_pred, y_vae, z_mean, z_var = model(x_batch)
            loss = compute_myrnenko_loss(x_batch, y_batch, y_pred, y_vae, z_mean, z_var)
            loss += sum(model.losses)

            val_loss.update_state(loss)

        avg_val_loss = val_loss.result() / n_val
        val_loss.reset_states()
        print(f'Validation loss: {avg_val_loss}.')

        # Write logs.
        if args.log_file:
            with open(args.log, 'w') as f:
                print(f'{epoch},{avg_train_loss},{avg_val_loss}\n', file=f)


if __name__ == '__main__':
    args = train_parser()
    main(args)
