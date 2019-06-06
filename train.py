"""Contains training loops."""
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.losses import DiceLoss, CustomLoss
from utils.metrics import DiceCoefficient
from utils.optimizer import ScheduledAdam
from utils.constants import *
from utils.utils import prepare_dataset, prepare_batch, prepare_val_set
from model.volumetric_cnn import VolumetricCNN


def prepare_batch(X, y, prob=0.5, data_format='channels_last'):
    """Performs augmentation and cropping in training."""
    # Data augmentation.
    if np.random.uniform() < prob:
        axes = (1, 2, 3) if data_format == 'chanenls_last' else (2, 3, 4)
        X = tf.reverse(X, axis=axes)
        y = tf.reverse(y, axis=axes)

    # Sample corner points.
    h = np.random.randint(H, RAW_H)
    w = np.random.randint(W, RAW_W)
    d = np.random.randint(D, RAW_D)

    if data_format == 'channels_last':
        X = X[:, h-H:h, w-W:w, d-D:d, :]
        y = y[:, h-H:h, w-W:w, d-D:d, :]
    else:
        X = X[:, :, h-H:h, w-W:w, d-D:d]
        y = y[:, :, h-H:h, w-W:w, d-D:d]
        
    return X, y


def prepare_val_set(dataset, n_sets=2, prob=0.5, data_format='channels_last'):
    """Prepares validation sets (with cropping and flipping)."""
    def parse_example(X, y):
        return prepare_batch(X, y, prob=prob, data_format=data_format)
    
    for i in range(n_sets):
        if i == 0: val_set = dataset.map(parse_example)
        else: val_set = val_set.concatenate(dataset.map(parse_example))

    return val_set


def prepare_dataset(path, batch_size, buffer_size=1000, data_format='channels_last', repeat=False):
    """Returns a BatchDataset object containing loaded data."""
    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, example_desc)
        if data_format == 'channels_last':
            X = tf.reshape(parsed['X'], (RAW_H, RAW_W, RAW_D, IN_CH))
            y = tf.reshape(parsed['y'], (RAW_H, RAW_W, RAW_D, 1))
            y = tf.squeeze(tf.cast(y, tf.int32), axis=-1)
            y = tf.one_hot(y, OUT_CH, axis=-1, dtype=tf.float32)
            y = y[:, :, :, 1:]
        elif data_format == 'channels_first':
            X = tf.reshape(parsed['X'], (IN_CH, RAW_H, RAW_W, RAW_D))
            y = tf.reshape(parsed['y'], (1, RAW_H, RAW_W, RAW_D))
            y = tf.squeeze(tf.cast(y, tf.int32), axis=0)
            y = tf.one_hot(y, OUT_CH, axis=0, dtype=tf.float32)
            y = y[1:, :, :, :]

        return (X, y)

    def get_dataset_len(tf_dataset):
        """Returns length of dataset until tf.data.experimental.cardinality is fixed."""
        # return tf.data.experimental.cardinality(tf_dataset)
        return sum(1 for _ in tf_dataset)

    example_desc = {
        'X': tf.io.FixedLenFeature([RAW_H * RAW_W * RAW_D * IN_CH], tf.float32),
        'y': tf.io.FixedLenFeature([RAW_H * RAW_W * RAW_D * 1], tf.float32)
    }

    dataset = tf.data.TFRecordDataset(path)
    dataset_len = get_dataset_len(dataset)

    if repeat:
        dataset = (dataset.map(parse_example)
                      .repeat()
                      .shuffle(buffer_size)
                      .batch(batch_size))
    else:
        dataset = (dataset.map(parse_example)
                      .shuffle(buffer_size)
                      .batch(batch_size))

    return dataset, dataset_len


def custom_train(args):
    # Load data.
    train_data, n_train = prepare_dataset(args.train_loc, args.batch_size,
                                          buffer_size=100, data_format=args.data_format, repeat=False)
    val_data, n_val = prepare_dataset(args.val_loc, args.batch_size,
                                      buffer_size=50, data_format=args.data_format, repeat=False)
    val_data = prepare_val_set(val_data, n_sets=args.n_val_sets,
                                prob=args.mirror_prob, data_format=args.data_format)
    n_val *= args.n_val_sets
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))

    # Initialize model.
    model = VolumetricCNN(
                    data_format=args.data_format,
                    kernel_size=args.conv_kernel_size,
                    groups=args.gn_groups,
                    reduction=args.se_reduction,
                    use_se=args.use_se,
                    kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale),
                    kernel_initializer=args.kernel_init,
                    downsampling=args.downsamp,
                    upsampling=args.upsamp,
                    normalization=args.norm)

    # Build model with initial forward pass.
    _ = model(tf.zeros(shape=(1, H, W, D, IN_CH) if args.data_format == 'channels_last' \
                                    else (1, IN_CH, H, W, D)))

    # Get starting epoch.
    start_epoch = model.epoch.value().numpy()
    
    # Load weights if specified.
    if args.load_file:
        model.load_weights(args.load_file)
    n_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
    print('Total number of parameters: {}'.format(n_params))

    optimizer = ScheduledAdam(learning_rate=args.lr)

    # Initialize loss and metrics.
    loss_fn = CustomLoss(decoder_loss=args.decoder_loss, data_format=args.data_format)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accu = tf.keras.metrics.BinaryAccuracy(name='train_accu')
    train_prec = tf.keras.metrics.Precision(name='train_prec')
    train_reca = tf.keras.metrics.Recall(name='train_reca')
    train_dice = DiceCoefficient(name='train_dice', data_format=args.data_format)

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accu = tf.keras.metrics.BinaryAccuracy(name='val_accu')
    val_prec = tf.keras.metrics.Precision(name='val_prec')
    val_reca = tf.keras.metrics.Recall(name='val_reca')
    val_dice = DiceCoefficient(name='val_dice', data_format=args.data_format)

    # Load model weights if specified.
    if args.load_file:
        model.load_weights(args.load_file)
    
    # Set up logging.
    if args.log_file:
        with open(args.log_file, 'w') as f:
            header = ','.join(['epoch',
                               'lr',
                               'train_loss',
                               'train_accu',
                               'train_prec',
                               'train_reca',
                               'train_dice',
                               'val_loss',
                               'val_accu',
                               'val_prec',
                               'val_reca',
                               'val_dice'])
            f.write(header + '\n')

    best_val_loss = float('inf')
    patience = 0

    # Train.
    for epoch in range(start_epoch, args.n_epochs):
        print('Epoch {}.'.format(epoch))
        model.epoch.assign(epoch)           # Track epoch in model.
        optimizer(epoch=epoch)              # Schedule learning rate.

        # Training epoch.
        for step, (X, y) in tqdm(enumerate(train_data, 1), total=n_train, desc='Training      '):
            X, y = prepare_batch(X, y, prob=args.mirror_prob, data_format=args.data_format)
            with tf.device(args.device):
                with tf.GradientTape() as tape:
                    # Forward and loss.
                    y_pred, y_vae, z_mean, z_logvar = model(X, training=True)
                    loss = loss_fn(X, y, y_pred, y_vae, z_mean, z_logvar)
                    loss += tf.reduce_sum(model.losses)

                # Gradients and backward.
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.update_state(loss)
                train_accu.update_state(y, y_pred)
                train_prec.update_state(y, y_pred)
                train_reca.update_state(y, y_pred)
                train_dice.update_state(y, y_pred)

            # Output loss to console.
            if args.log_steps > 0 and step % args.log_steps == 0:
                print('Step {}. Loss: {l: .4f}, Accu: {v: 1.4f}, Prec: {p: 1.4f}, \
                       Reca: {r: 1.4f}, Dice: {d: 1.4f}.'
                       .format(step, l=train_loss.result(),
                                     v=train_accu.result(),
                                     p=train_prec.result(),
                                     r=train_reca.result(),
                                     d=train_dice.result()), flush=True)

        print('Training. Loss: {l: .4f}, Accu: {v: 1.4f}, Prec: {p: 1.4f}, \
               Reca: {r: 1.4f}, Dice: {d: 1.4f}.'
                       .format(l=train_loss.result(),
                               v=train_accu.result(),
                               p=train_prec.result(),
                               r=train_reca.result(),
                               d=train_dice.result()), flush=True)

        # Validation epoch.
        for step, (X, y) in tqdm(enumerate(val_data), total=n_val, desc='Validation    '):
            with tf.device(args.device):
                # Forward and loss.
                y_pred, y_vae, z_mean, z_logvar = model(X, training=False)
                loss = loss_fn(X, y, y_pred, y_vae, z_mean, z_logvar)
                loss += tf.reduce_sum(model.losses)

                val_loss.update_state(loss)
                val_accu.update_state(y, y_pred)
                val_prec.update_state(y, y_pred)
                val_reca.update_state(y, y_pred)
                val_dice.update_state(y, y_pred)

        print('Validation. Loss: {l: .4f}, Accu: {v: 1.4f}, Prec: {p: 1.4f}, \
               Reca: {r: 1.4f}, Dice: {d: 1.4f}.'
                       .format(l=val_loss.result(),
                               v=val_accu.result(),
                               p=val_prec.result(),
                               r=val_reca.result(),
                               d=val_dice.result()), flush=True)

        # Write logs.
        if args.log_file:
            with open(args.log_file, 'a') as f:
                entry = ','.join([str(epoch),
                                  str(optimizer.learning_rate.numpy()),
                                  str(train_loss.result().numpy()),
                                  str(train_accu.result().numpy()),
                                  str(train_prec.result().numpy()),
                                  str(train_reca.result().numpy()),
                                  str(train_dice.result().numpy()),
                                  str(val_loss.result().numpy()),
                                  str(val_prec.result().numpy()),
                                  str(val_reca.result().numpy()),
                                  str(val_accu.result().numpy()),
                                  str(val_dice.result().numpy())])
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
        train_accu.reset_states()
        train_prec.reset_states()
        train_reca.reset_states()
        train_dice.reset_states()
        val_loss.reset_states()
        val_accu.reset_states()
        val_prec.reset_states()
        val_reca.reset_states()
        val_dice.reset_states()


if __name__ == '__main__':
    args = train_parser()
    print('Train args: {}'.format(args), flush=True)
    custom_train(args)
