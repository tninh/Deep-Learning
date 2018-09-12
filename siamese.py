import random

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
import os


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class SiameseFaceNet(object):
    model_name = 'siamese-face-net'

    def __init__(self):
        self.model = None
        self.vgg16_include_top = False

        self.labels = None
        self.config = None
        self.input_shape = None
        self.threshold = 0.5
        self.vgg16_model = None

    def encoding_images(self, image_path):
        print('We are encoding: ', image_path)
        if self.vgg16_model is None:
            self.vgg16_model = self.create_vgg16_model()

        image = cv2.imread(image_path, 1)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        return self.vgg16_model.predict(input)

    def load_model(self, model_dir_path):
        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        self.config = np.load(config_file_path).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        self.threshold = self.config['threshold']
        self.vgg16_include_top = self.config['vgg16_include_top']
        self.model = self.create_network(input_shape=self.input_shape)
        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    def accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < self.threshold, y_true.dtype)))

    def create_network(self, input_shape):
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        rms = RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[self.accuracy])

        print(model.summary())

        return model

    def create_pairs(self, database, names):
        num_classes = len(database)
        pairs = []
        labels = []
        n = min([len(database[name]) for name in database.keys()])
        for d in range(len(names)):
            name = names[d]
            x = database[name]
            for i in range(n):
                pairs += [[x[i], x[(i + 1) % n]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                x1, x2 = x[i], database[names[dn]][i]
                pairs += [[x1, x2]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def create_pairs_1(self, database, names):
        num_classes = len(database)
        pairs = []
        labels = []
        n = min([len(database[name]) for name in database.keys()])
        for d in range(len(names)):
            name = names[d]
            x = database[name][0]
            for i in range(n):
                pairs += [[x[i], x[(i + 1) % n]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                x1, x2 = x[i], database[names[dn]][0][i]
                pairs += [[z1, z2]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)


    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-architecture.h5'

    def create_vgg16_model(self):
        base_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        x = base_model.output
        base_model = VGG16(include_top=True, weights='imagenet')
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1020, activation='relu')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
                layer.trainable = False
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, database, model_dir_path, epochs=None, batch_size=None, threshold=None, vgg16_include_top=None):
        if threshold is not None:
            self.threshold = threshold
        if batch_size is None:
            batch_size = 128
        if epochs is None:
            epochs = 20
        if vgg16_include_top is not None:
            self.vgg16_include_top = vgg16_include_top

        for name, feature in database.items():
            self.input_shape = feature[0].shape
            break

        self.model = self.create_network(input_shape=self.input_shape)
        architecture_file_path = self.get_architecture_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        names = []
        self.labels = dict()
        for name in database.keys():
            names.append(name)
            self.labels[name] = len(self.labels)

        self.config = dict()
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold
        self.config['vgg16_include_top'] = self.vgg16_include_top

        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)

        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        x_1, x_2 = self.create_pairs(database, names)

        #x_1, x_2 = self.create_pairs_1(database, names)
        print('data set pairs: ', x_1.shape)

        self.model.fit([x_1[:, 0], x_1[:, 1]], x_2,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2,
                       verbose=1,
                       callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)

    def face_recognition(self, image_path, database, threshold=None):
        if threshold is not None:
            self.threshold = threshold
        encoding = self.encoding_images(image_path)
       
        min_dist = 100
        identity = None
       
        for (name, images) in database.items():
            input_pairs = []
            for i in range(len(images)):
                input_pairs.append([encoding, images[i]])
            input_pairs = np.array(input_pairs)
            distance = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

            print("For " + str(name) + ", the distance is " + str(distance))
            
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > self.threshold:
            print("Can't find the image")
        else:
            print("May it's" + str(identity) + ", error is:" + str(min_dist))
        return min_dist, identity


def main():
    net = SiameseFaceNet()
    net.vgg16_include_top = True


if __name__ == '__main__':
    main()
