import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.arg_parser import train_parser
from utils.loss import compute_myrnenko_loss
from model.volumetric_seq2seq import VolumetricSeq2Seq


def prepare_data(path, batch_size, seed=2021):
    data = np.load(path)
    data = (tf.data.Dataset.from_tensor_slices((data['X'], data['y']))
                           .shuffle(seed)
                           .batch(batch_size))
    return data


def main(args):
    # Load data.
    train_data = prepare_data(args.train_loc, args.batch_size, seed=2021)
    val_data = prepare_data(args.val_loc, args.batch_size, seed=2021)

    # Set up.
    model = VolumetricSeq2Seq(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        dropout=args.dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    n_train = len(train_data)
    n_val = len(val_data)

    if args.log:
        with open(args.log, 'w') as f:
            print(f'epoch,train_loss,val_loss\n', file=f)

    # Train.
    for epoch in range(args.n_epochs):
        print(f'Epoch {epoch}.')

        # Training epoch.
        for step, (x_batch, y_batch) in tqdm(enumerate(train_data)):
            with tf.GradientTape() as tape:
                y_pred, y_vae, z_mean, z_var = model(x_batch)
                
                loss = compute_myrnenko_loss(x_batch, y_batch, y_pred, y_vae, z_mean, z_var)
                loss += sum(model.losses)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)

        avg_train_loss = train_loss.result() / n_train
        train_loss.reset_states()
        print(f'Training loss: {avg_train_loss}.')

        # Validation epoch.
        for step, (x_batch, y_batch) in tqdm(enumerate(val_data)):
            y_pred, y_vae, z_mean, z_var = model(x_batch)

            loss = compute_myrnenko_loss(x_batch, y_batch, y_pred, y_vae, z_mean, z_var)
            loss += sum(model.losses)

            val_loss.update_state(loss)

        avg_val_loss = val_loss.result() / n_val
        val_loss.reset_states()
        print(f'Validation loss: {avg_val_loss}.')

        # Write logs.
        if args.log:
            with open(args.log, 'w') as f:
                print(f'{epoch},{avg_train_loss},{avg_val_loss}\n', file=f)


if __name__ == '__main__':
    args = train_parser()
    main(args)
