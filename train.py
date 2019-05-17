import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.loss import myrnenko_loss
from utils.optimizer import ScheduledAdam
from utils.constants import *
from utils.metrics import dice_coefficient, segmentation_accuracy
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
    dataset = dataset.map(parse_example).shuffle(10000).batch(batch_size)

    return dataset, get_dataset_len(dataset)


def evaluate(x, y_true, y_pred, y_vae, z_mean, z_logvar, data_format='channels_last'):
    loss = myrnenko_loss(x, y_true, y_pred, y_vae, z_mean,z_logvar, data_format=data_format)
    voxel_accu = segmentation_accuracy(y_pred, y_true, data_format=data_format)
    dice_coeff = dice_coefficient(y_pred, y_true, data_format=data_format)

    return loss, voxel_accu, dice_coeff


def main(args):
    # Load data.
    train_data, n_train = prepare_dataset(args.train_loc, args.batch_size)
    val_data, n_val = prepare_dataset(args.val_loc, args.batch_size)
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))

    # Initialize model and optimizer
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        dropout=args.dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))
    optimizer = ScheduledAdam(learning_rate=args.lr)

    # Initialize loss, accuracies, and dice coefficients.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_voxel_accu = tf.keras.metrics.Mean(name='train_voxel_accu')
    train_dice_coeff = tf.keras.metrics.Mean(name='train_dice_coeff')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_voxel_accu = tf.keras.metrics.Mean(name='val_voxel_accu')
    val_dice_coeff = tf.keras.metrics.Mean(name='val_dice_coeff')

    # Load model weights if specified.
    if args.load_file:
        model.load_weights(args.load_file)
    
    # Set up logging.
    if args.log_file:
        with open(args.log_file, 'w') as f:
            header = ','.join(['epoch',
                               'lr',
                               'train_loss',
                               'train_voxel_accu',
                               'train_dice_coeff',
                               'val_loss',
                               'val_voxel_accu',
                               'val_dice_coeff'])
            f.write(header + '\n')

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
                    loss, voxel_accu, dice_coeff = evaluate(
                                                        x_batch, y_batch, y_pred,
                                                        y_vae, z_mean, z_logvar,
                                                        data_format=args.data_format)
                    loss += sum(model.losses)

                # Gradients and backward.
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.update_state(loss)
                train_voxel_accu.update_state(voxel_accu)
                train_dice_coeff.update_state(dice_coeff)

            # Output loss to console.
            if step % args.log_steps == 0:
                print('Step {}. Loss: {l: .4f}, Voxel Accu: {v: 3.3f}, Dice Coeff: {d: 0.4f}.'
                       .format(step, l=train_loss.result(),
                                     v=train_voxel_accu.result()*100,
                                     d=train_dice_coeff.result()))

        print('Training. Loss: {l: .4f}, Voxel Accu: {v: 3.3f}, Dice Coeff: {d: 0.4f}.'
               .format(l=train_loss.result(),
                       v=train_voxel_accu.result()*100,
                       d=train_dice_coeff.result()))

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
                loss, voxel_accu, dice_coeff = evaluate(
                                                    x_batch, y_batch, y_pred,
                                                    y_vae, z_mean, z_logvar,
                                                    data_format=args.data_format)
                loss += sum(model.losses)

                val_loss.update_state(loss)
                val_voxel_accu.update_state(voxel_accu)
                val_dice_coeff.update_state(dice_coeff)

        print('Validation. Loss: {l: .4f}, Voxel Accu: {v: 3.3f}, Dice Coeff: {d: 0.4f}.'
               .format(l=val_loss.result(),
                       v=val_voxel_accu.result()*100,
                       d=val_dice_coeff.result()))

        # Write logs.
        if args.log_file:
            with open(args.log_file, 'a') as f:
                entry = ','.join([str(epoch),
                                  str(optimizer.learning_rate.numpy()),
                                  str(train_loss.result().numpy()),
                                  str(train_voxel_accu.result().numpy()),
                                  str(train_dice_coeff.result().numpy()),
                                  str(val_loss.result().numpy()),
                                  str(val_voxel_accu.result().numpy()),
                                  str(val_dice_coeff.result().numpy())])
                f.write(entry + '\n')

        # Checkpoint and patience.
        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            model.save_weights(args.save_file)
            patience = 0
            print('Saved model weights.')
        elif patience == args.patience:
            print('Validation loss has not improved in {} epochs. Stopped training.'
                    .format(args.patience))
            break
        else:
            patience += 1

        # Reset statistics.
        train_loss.reset_states()
        train_voxel_accu.reset_states()
        train_dice_coeff.reset_states()
        val_loss.reset_states()
        val_voxel_accu.reset_states()
        val_dice_coeff.reset_states()


if __name__ == '__main__':
    args = train_parser()
    main(args)
