from models import ModelGenerator
from dataset_master import DSMaster
import tensorboard_master

from tensorboard_master import HP_CONV_CORE, HP_LR, HP_OPTIMIZER, HP_IMAGE_SIZE
import tensorflow as tf
import time

DATA_SET_PATH = r"D:\datasets\net\chest_xray"

LOGDIR = "logs\\fit\\ascent3DCov_maxbatchcnn_real\\"


def main():

    run = 0

    train_gens = {}
    val_gens = {}

    for image_size in HP_IMAGE_SIZE.domain.values:
        ds_master = DSMaster(DATA_SET_PATH, image_size)
        print("Form acsent train set")
        train_gens[image_size] = ds_master.get_ascent_train_gen()
        print("Form acsent test set")
        val_gens[image_size] = ds_master.get_ascent_test_gen(image_num=image_size * 2)
        # print("Get original train set")
        # train_gens[image_size] = ds_master.get_raw_train_gen()
        # print("Get original test set")
        # val_gens[image_size] = ds_master.get_raw_validation_gen()

    for image_size in HP_IMAGE_SIZE.domain.values:
        for conv_core in HP_CONV_CORE.domain.values:
            for learning_rate in HP_LR.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    end_log_dir = LOGDIR + "run" + str(run)
                    with tf.summary.create_file_writer(end_log_dir).as_default():
                        print("run:", run)
                        hparams = {
                            HP_CONV_CORE: conv_core,
                            HP_LR: learning_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_IMAGE_SIZE: image_size
                        }

                        tensorboard_master.record_params(hparams)
                        print("Callback forming")

                        tensorboard_callback = tensorboard_master.default_callback(end_log_dir)
                        # hyperparams_callback = tensorboard_master.hyperparams_callback(end_log_dir, hparams)

                        ds_master = DSMaster(DATA_SET_PATH, image_size)

                        train_gen = train_gens[image_size]
                        test_gen = val_gens[image_size]

                        model = ModelGenerator.base_cnn_maxpool_batch_normalization(hparams)
                        print("Training start", hparams.values())
                        start = time.time()
                        history = model.fit(train_gen,
                                            epochs=400,
                                            validation_data=test_gen,
                                            verbose=2,
                                            callbacks=[tensorboard_callback], shuffle=False)
                        tensorboard_master.record_scalar(time.time() - start, "train_time")

                        # tensorboard_master.record_scalar(history.history["val_accuracy"][0], "val_accuracy")
                        run = run + 1


if __name__ == "__main__":
    main()
