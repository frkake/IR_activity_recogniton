"""
Train our RNN on bottlecap or prediction files generated from our CNN.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)
np.random.seed(0)


def train(data_type, seq_length, model, saved_model=None,
          concat=False, class_limit=None, image_shape=None,
          load_to_memory=False):
    # Set variables.
    nb_epoch = 1000
    batch_size = 16

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '-' + data_type + \
                 '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    incepcheck = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '-' + data_type + \
                 '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir='./data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
                           str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type, concat)
        X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', data_type, concat)

    elif model == 'div_crnn':
        generator = data.frame_generator2(batch_size, 'train', data_type, concat)
        val_generator = data.frame_generator2(batch_size, 'test', data_type, concat)

    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, concat)
        val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # model_json_str = rm.model.to_json()
    # open('/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/checkpoints/json_model.json','w').write(model_json_str)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[checkpointer, tb, csv_logger],
            epochs=nb_epoch)
    # elif model == 'inception*':
    #     rm.model.fit_generator(
    #         generator=generator,
    #         steps_per_epoch=steps_per_epoch,
    #         epochs=nb_epoch,
    #         verbose=1,
    #         callbacks=[incepcheck, tb, csv_logger],
    #         validation_data=val_generator,
    #         validation_steps=10)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer, tb, csv_logger],
            validation_data=val_generator,
            validation_steps=10)


def main():
    """These are the main training settings. Set each before running
    this file."""
    model = "div_crnn"  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 10
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'mlp' or model == 'lstm':
        data_type = 'features'
        image_shape = None

    else:
        data_type = 'images'
        image_shape = (16, 16, 1)
        load_to_memory = False

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, concat=concat, image_shape=image_shape,
          load_to_memory=load_to_memory)


if __name__ == '__main__':
    main()
