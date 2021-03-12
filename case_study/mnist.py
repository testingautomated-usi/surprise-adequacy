import os
import shutil
import time
from typing import Dict

import foolbox
import numpy as np
import tensorflow as tf
import uncertainty_wizard as uwiz

from surprise.surprise_adequacy import SurpriseAdequacyConfig
from case_study import config, utils



NUM_MODELS = 100


class TrainContext(uwiz.models.ensemble_utils.DeviceAllocatorContextManager):

    @classmethod
    def file_path(cls) -> str:
        return "temp-ensemble.txt"

    @classmethod
    def run_on_cpu(cls) -> bool:
        return False

    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        return {
            0: 4,
            1: 4
        }

    @classmethod
    def gpu_memory_limit(cls) -> int:
        return 1000


class EvalContext(TrainContext):
    @classmethod
    def virtual_devices_per_gpu(cls) -> Dict[int, int]:
        return {
            0: 2,
            1: 2
        }


def prepare_adv_data(model):
    x_train, _, x_test, y_test = _get_dataset()
    badge_size = 100
    x_test = np.reshape(x_test, (-1, badge_size, 28, 28, 1))
    y_test = np.reshape(y_test, (-1, badge_size))

    adv = []
    for i in range(x_test.shape[0]):
        fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 1), device='CPU:0')
        attack = foolbox.attacks.LinfFastGradientAttack()
        attack_x = tf.convert_to_tensor(x_test[i])
        attack_y = tf.convert_to_tensor(y_test[i], dtype=tf.int32)
        advs, _, success = attack(fmodel, attack_x, attack_y, epsilons=[0.5])
        adv.append(advs)
    return np.concatenate(adv).reshape((-1, 28, 28, 1))


def run_time_experiments(model_id, model):
    if model_id > 0:
        print("time experiments are performed only on the first model")
    x_train, _, x_test, _ = _get_dataset()
    temp_folder = "/tmp/" + str(time.time())
    os.mkdir(temp_folder)
    sa_config = SurpriseAdequacyConfig(saved_path=temp_folder, is_classification=True, layer_names=["last_dense"],
                                       ds_name=f"mnist_{model_id}", num_classes=10)
    utils.time_experiments(model=model,
                           sa_config=sa_config,
                           train_x=x_train,
                           test_x=x_test)


def run_experiments(model_id, model):
    if model_id < 10:
        return None
    x_train, _, x_test, y_test = _get_dataset()
    advs = prepare_adv_data(model)
    corrupted = np.load(f"{config.DATASETS_BASE_FOLDER}mnist_corrupted.npy") / 255.
    test_data = {
        'nominal': (x_test, y_test),
        'adv_fga_0.5': (advs, y_test),
        'corrupted': (corrupted, y_test),
    }
    temp_folder = "/tmp/" + str(time.time())
    os.mkdir(temp_folder)
    sa_config = SurpriseAdequacyConfig(saved_path=temp_folder, is_classification=True, layer_names=["last_dense"],
                                       ds_name=f"mnist_{model_id}", num_classes=10)
    results = utils.run_experiments(model=model,
                                    train_x=x_train,
                                    test_data=test_data,
                                    sa_config=sa_config)
    utils.save_results_to_fs(results=results, case_study="mnist", model_id=model_id)
    shutil.rmtree(temp_folder)


def train_model(model_id):
    """
    Trains an mnist model. According to https://keras.io/examples/vision/mnist_convnet/, but with an additional layer.
    :param model_id:
    :return:
    """
    import tensorflow as tf
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, name="last_dense"),
            tf.keras.layers.Dense(10, activation="softmax", name="sm_output"),
        ]
    )

    x_train, y_train, _, _ = _get_dataset()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

    return model, history.history


def _get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    #
    # OOD Detection Capability Experiments
    #
    tf.keras.datasets.mnist.load_data()

    model_collection = uwiz.models.LazyEnsemble(num_models=NUM_MODELS,
                                                model_save_path=config.MODELS_BASE_FOLDER + "mnist",
                                                delete_existing=False,
                                                expect_model=True)
    histories = model_collection.create(
        train_model, num_processes=8, context=TrainContext
    )

    model_collection.consume(
        run_experiments, num_processes=0,
    )

    #
    # Runtime experiments
    #
    single_model_ensemble = uwiz.models.LazyEnsemble(num_models=1,
                                                     model_save_path=config.MODELS_BASE_FOLDER + "mnist",
                                                     delete_existing=False,
                                                     expect_model=True)
    single_model_ensemble.consume(run_time_experiments, num_processes=0)
