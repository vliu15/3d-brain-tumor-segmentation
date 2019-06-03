"""Contains training loops."""
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.losses import DiceLoss, CustomLoss
from utils.metrics import DiceCoefficient
from utils.optimizer import ScheduledAdam
from utils.constants import *
from utils.utils import prepare_dataset
from model.volumetric_cnn import VolumetricCNN


def custom_train(args):
    # Load data.
    train_data, n_train = prepare_dataset(
                            args.train_loc, args.batch_size, buffer_size=100, data_format=args.data_format, repeat=False)
    val_data, n_val = prepare_dataset(
                            args.val_loc, args.batch_size, buffer_size=50, data_format=args.data_format, repeat=False)
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
    _ = model(tf.zeros(shape=[1] + list(CHANNELS_LAST_X_SHAPE) if args.data_format == 'channels_last'
                                    else [1] + list(CHANNELS_FIRST_X_SHAPE)))

    # Get starting epoch.
    start_epoch = model.epoch.value().numpy()
    
    # Load weights if specified.
    if args.load_file:
        model.load_weights(args.load_file)
    n_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
    print('Total number of parameters: {}'.format(n_params))

    optimizer = ScheduledAdam(learning_rate=args.lr)

    # Initialize loss and metrics.
    # loss_fn = DiceLoss(data_format=args.data_format)
    loss_fn = CustomLoss(data_format=args.data_format)

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

        # Track epoch in model.
        model.epoch.assign(epoch)

        # Schedule learning rate.
        optimizer(epoch=epoch)

        # Training epoch.
        for step, (X, y) in tqdm(enumerate(train_data, 1), total=n_train, desc='Training      '):
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


def keras_train(args):
    # Load data.
    train_data, n_train = prepare_dataset(
                            args.train_loc, args.batch_size, buffer_size=100, data_format=args.data_format, repeat=True)
    val_data, n_val = prepare_dataset(
                            args.val_loc, args.batch_size, buffer_size=50, data_format=args.data_format, repeat=True)
    print('{} training examples.'.format(n_train))
    print('{} validation examples.'.format(n_val))

    with tf.device(args.device):
        if args.load_file:
            model = tf.keras.models.load_model(args.load_file)
        else:
            model = VolumetricCNN(data_format=args.data_format,
                                  kernel_size=args.conv_kernel_size,
                                  groups=args.gn_groups,
                                  reduction=args.se_reduction,
                                  use_se=args.use_se,
                                  kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale),
                                  kernel_initializer=args.kernel_init,
                                  downsampling=args.downsamp,
                                  upsampling=args.upsamp,
                                  normalization=args.norm)

        model.compile(optimizer=tf.keras.optimizers.Adam(
                                    learning_rate=args.lr,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-7,
                                    amsgrad=False),
                      loss=DiceLoss(data_format=args.data_format),
                      metrics=[tf.keras.metrics.Accuracy(),
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(),
                               DiceCoefficient(data_format=args.data_format)])

        history = model.fit(train_data,
                            epochs=args.n_epochs,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(
                                            Scheduler(args.n_epochs, args.lr, args.warmup_epochs)),
                                       tf.keras.callbacks.ModelCheckpoint(
                                            args.save_file,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=False),
                                       tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            min_delta=1e-3,
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

    print('Train args: {}'.format(args), flush=True)

    custom_train(args)
    # history = keras_train(args)
