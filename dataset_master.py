from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import image_loader
from matplotlib import pyplot as plt


class DSMaster:
    def __init__(self, ds_path, image_size):
        self.ds_path = ds_path
        self.image_size = image_size

    def get_raw_train_gen(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.ds_path + "/train",
            color_mode="grayscale",
            target_size=(self.image_size, self.image_size),
            batch_size=32,
            class_mode='binary')
        return train_generator

    def get_raw_validation_gen(self):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow_from_directory(
            self.ds_path + "/test",
            color_mode="grayscale",
            target_size=(self.image_size, self.image_size),
            batch_size=32,
            class_mode='binary')
        return validation_generator

    def get_ascent_train_gen(self, image_num=None):
        train_dogs_set = image_loader.load_ascent_images(self.ds_path + "/train/dogs", self.image_size,
                                                         image_num=image_num)
        train_cats_set = image_loader.load_ascent_images(self.ds_path + "/train/cats", self.image_size,
                                                         image_num=image_num)
        train_generator = self.__get_gen_from_flow(train_dogs_set, train_cats_set)

        return train_generator

    def get_ascent_test_gen(self, image_num=None):

        offset = 0
        import glob
        testSize = len(glob.glob(self.ds_path + r"/test/dogs" + r"/*.jpg"))
        test_dogs_set = []
        test_cats_set = []

        while offset < testSize:
            test_dogs_ascent = image_loader.load_ascent_images(self.ds_path + "/test/dogs", self.image_size,
                                                               image_num=image_num, offset=offset)
            if len(test_dogs_set) == 0:
                test_dogs_set = test_dogs_ascent
            else:
                test_dogs_set = test_dogs_set + test_dogs_ascent

            test_cats_ascent = image_loader.load_ascent_images(self.ds_path + "/test/cats", self.image_size,
                                                               image_num=image_num, offset=offset)
            if len(test_cats_set) == 0:
                test_cats_set = test_cats_ascent
            else:
                test_cats_set = test_cats_set + test_cats_ascent
            offset += image_num

        validation_generator = self.__get_gen_from_flow(test_dogs_set, test_cats_set)

        return validation_generator

    def get_inner_ascent_train_gen(self, image_num=None):
        train_dogs_set = image_loader.load_inner_ascent_images(self.ds_path + "/train/dogs", self.image_size,
                                                               image_num=image_num)
        train_cats_set = image_loader.load_inner_ascent_images(self.ds_path + "/train/cats", self.image_size,
                                                               image_num=image_num)
        print("Train Set of ", len(train_dogs_set), "dogs and ", len(train_cats_set), "cats")

        train_generator = self.__get_gen_from_flow(train_dogs_set, train_cats_set)

        return train_generator

    def get_inner_ascent_test_gen(self, image_num=None):
        test_dogs_set = image_loader.load_inner_ascent_images(self.ds_path + "/test/dogs", self.image_size,
                                                              image_num=image_num)
        test_cats_set = image_loader.load_inner_ascent_images(self.ds_path + "/test/cats", self.image_size,
                                                              image_num=image_num)
        print("Test Set of ", len(test_dogs_set), "dogs and ", len(test_cats_set), "cats")

        validation_generator = self.__get_gen_from_flow(test_dogs_set, test_cats_set)

        return validation_generator

    def __get_gen_from_flow(self, set1, set2):
        all_set = np.concatenate([set1, set2])

        labels1 = np.zeros((np.shape(set1)[0], 1))
        labels2 = np.ones((np.shape(set2)[0], 1))
        target = np.concatenate([labels1, labels2])

        generator = ImageDataGenerator().flow(
            x=all_set,
            y=target,
            batch_size=8)

        return generator
