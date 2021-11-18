import os,sys,argparse
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D, Conv2D, ReLU
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

EFN_MODEL = {
    'b0': EfficientNetB0,
    'b1': EfficientNetB1,
    'b2': EfficientNetB2,
    'b3': EfficientNetB3,
    'b4': EfficientNetB4,
    'b5': EfficientNetB5,
    'b6': EfficientNetB6,
    'b7': EfficientNetB7
}

EFN_INPUT_SHAPE = {
    'b0': (224,224,3),
    'b1': (240,240,3),
    'b2': (260,260,3),
    'b3': (300,300,3),
    'b4': (380,380,3),
    'b5': (456,456,3),
    'b6': (528,528,3),
    'b7': (600,600,3)
}

class TrainableModel(object):
    def fit(self, x_train, y_train, x_test=None, y_test=None, batch_size=128, epochs=1, lr=0.001, momentum=0.9, loss='categorical_crossentropy', callbacks=None, data_augmentation=True, metrics=['accuracy']):
        #self.model.compile(optimizer=SGD(lr=lr, momentum=momentum), loss=loss, metrics=metrics)
        self.model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
        datagen = ImageDataGenerator(**datagen_params) if data_augmentation else ImageDataGenerator()
        return self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test) if x_test is not None else None, callbacks=callbacks)

    def fit_dir(self, dir, batch_size=128, epochs=1, lr=0.001, momentum=0.9, loss='categorical_crossentropy', callbacks=None, data_aug=True):
        train_dir = os.path.join(dir, 'train') if os.path.exists(os.path.join(dir, 'train')) else dir
        test_dir = os.path.join(dir, 'val') if os.path.exists(os.path.join(dir, 'val')) else None
        train_generator = get_dir_generator(train_dir, self.input_shape[:2], batch_size=batch_size, train=data_aug)
        test_generator = get_dir_generator(test_dir, self.input_shape[:2], train=False, batch_size=batch_size) if test_dir is not None else None
        return self.fit_generator(train_generator, test_generator, epochs, lr, momentum, loss, callbacks)

    def fit_generator(self, train_gen, val_gen=None, epochs=1, lr=0.001, momentum=0.9, loss='categorical_crossentropy', callbacks=None, metrics=['accuracy']):
        #self.model.compile(optimizer=SGD(lr=lr, momentum=momentum), loss=loss, metrics=metrics)
        self.model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
        return self.model.fit(train_gen, steps_per_epoch=len(train_gen), epochs=epochs, verbose=2, validation_data=val_gen, validation_steps=len(val_gen), callbacks=callbacks, workers=4)

class Efficientnet(TrainableModel):
    def __init__(self, model_name='b0', weights='imagenet', activation=None):
        self.model_name = model_name
        self.input_shape = EFN_INPUT_SHAPE[model_name]
        print('model name: {}'.format(self.model_name))
        print('input size: {}'.format(self.input_shape))

        #load base model
        if activation is None:
            self.base_model = EFN_MODEL[model_name](include_top=False, weights=weights, input_shape=self.input_shape)
        else:
            kwargs = dict(activation=activation)
            self.base_model = EFN_MODEL[model_name](include_top=False, weights=weights, input_shape=self.input_shape, **kwargs)
        print('[efn] loaded base model')

    def _freeze_feature_extractor(self, BN=True):
        print('[info] freeze feature extractor layers (BN: {})'.format(BN))
        for layer in self.feature_extractor.layers:
            if BN or not isinstance(layer, BatchNormalization):
                layer.trainable = False

    def _unfreeze_feature_extractor(self):
        print('[info] unfreeze feature extractor layers')
        for layer in self.feature_extractor.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True

    def unfreeze_feature_extractor(self, layer_name):
        print('[info] unfreeze all feature extractor layers after {}'.format(layer_name))
        set_trainable = False
        for layer in self.feature_extractor.layers:
            if set_trainable:
                layer.trainable = True
                #print('[debug] {} is set to trainable'.format(layer.name))
            else:
                layer.trainable = False
            if layer.name == layer_name:
                set_trainable = True

    def save_weights(self, file_path='efn_weights.h5', include_top=True):
        if include_top:
            self.model.save_weights(file_path)
        else:
            self.feature_extractor.save_weights(file_path)

    def save(self, file_path, include_top=True):
        if include_top:
            self.model.save(file_path)
        else:
            self.feature_extractor.save(file_path)

    def load_weights(self, file_path, include_top=True):
        if include_top:
            self.model.load_weights(file_path)
        else:
            self.feature_extractor.load_weights(file_path)

class EfficientnetClassifier(Efficientnet):
    def __init__(self, num_classes, model_name='b0', weights='imagenet', activation=None):
        super(EfficientnetClassifier, self).__init__(model_name, weights, activation)

        self.num_classes = num_classes

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        self.feature_extractor = keras.models.Model(inputs=self.base_model.input, outputs=x)
        self.classifier = keras.models.Sequential([
            Dense(1024, input_shape=(self.feature_extractor.output_shape[1],)),
            #Activation('relu'),
            Dense(num_classes),
            Activation('softmax'),
        ])

        embeddings = self.feature_extractor.output
        predictions = self.classifier(embeddings)
        self.model = keras.models.Model(inputs=self.base_model.input, outputs=predictions)

        self._freeze_feature_extractor()
        #for layer in self.feature_extractor.layers[-17:]:
        #    print(layer.name)
        #    layer.trainable = True
        print('[efn] model initialized')
        #print(self.feature_extractor.summary())

    def test(self, dir, batch_size=128):
        if isinstance(dir, str):
            test_dir = os.path.join(dir, 'val') if os.path.exists(os.path.join(dir, 'val')) else dir
            datagen = get_dir_generator(test_dir, self.input_shape[:2], train=False, batch_size=batch_size)
        else:
            x_test, y_test = dir[0], dir[1]
            datagen = get_generator(x_test, y_test, train=False, batch_size=batch_size)
        confidence, true_label = [], []
        for num_batch, data in enumerate(datagen):
            x, y = data[0], data[1]
            output = self.model.predict(x)
            confidence.extend(output.copy())
            true_label.extend(y)
            if num_batch == len(datagen)-1: break
        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(true_label, confidence).numpy()
        print('loss: {}'.format(loss))
        true_labels = np.argmax(true_label, axis=1)
        preds = np.argmax(confidence, axis=1)
        acc = np.count_nonzero(true_labels==preds)/len(true_labels)
        print('acc: {}'.format(acc))
        cnf_mtx = sklearn.metrics.confusion_matrix(true_labels, preds)
        print('confusion matrix:')
        print(cnf_mtx.tolist())
        f1_score = sklearn.metrics.f1_score(true_labels, preds, average=None)
        print('f1 score: {}'.format(f1_score))
        return acc, loss, cnf_mtx, f1_score

    def extract_feature(self, datagen):
        embedding = []
        for batch_num, data in enumerate(datagen):
            x = data[0]
            emb = self.feature_extractor.predict(x)
            embedding.extend(emb.copy())
            if batch_num == len(datagen) - 1: break
        return embedding

    def get_confidence(self, datagen):
        conf, label = [], []
        for batch_num, data in enumerate(datagen):
            x, y = data[0], data[1]
            output = self.model.predict(x)
            conf.extend(output.copy())
            label.extend(np.argmax(y, axis=1))
            if batch_num == len(datagen) - 1: break
        return conf, label

class BottleNeck(TrainableModel):
    def __init__(self, latent_dim, input_shape):
        print('[info] init bottleneck model, latent_dim: {}, input_shape: {}'.format(latent_dim, input_shape))
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        encoder_layers = [tf.keras.layers.InputLayer(input_shape=input_shape)]
        input_size = input_shape[0]
        num_conv = 0
        filter_list = [input_shape[2]]
        while input_size%2==0:
            filter = 32*2**num_conv
            filter_list.append(filter)
            encoder_layers.append(tf.keras.layers.Conv2D(filters=filter, kernel_size=3, strides=(2, 2), activation='relu', name='bottleneck_conv_{}'.format(num_conv+1)))
            num_conv += 1
            input_size = input_size / 2
        encoder_layers.append(tf.keras.layers.Flatten(name='bottleneck_flatten'))
        encoder_layers.append(tf.keras.layers.Dense(latent_dim, name='bottleneck_dense_1'))
        self.encoder = tf.keras.Sequential(encoder_layers)

        conv_shape = (
            int(input_shape[0]/2**num_conv),
            int(input_shape[1]/2**num_conv),
            filter_list[-1])

        decoder_layers = [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=conv_shape[0]*conv_shape[1]*conv_shape[2], activation=tf.nn.relu, name='bottleneck_dense_2'),
            tf.keras.layers.Reshape(target_shape=conv_shape, name='bottleneck_reshape')
            ]
        for tconv_idx in range(num_conv):
            decoder_layers.append(tf.keras.layers.Conv2DTranspose(filters=filter_list[-(tconv_idx+2)], kernel_size=3, strides=(2, 2), padding="SAME", activation='relu', name='bottleneck_conv_trans_{}'.format(tconv_idx+1)))
        self.decoder = tf.keras.Sequential(decoder_layers)

        x = self.encoder.output
        x = self.decoder(x)

        self.model = tf.keras.models.Model(inputs=self.encoder.input, outputs=x)

        #print(self.model.summary())
        print('[info] bottleneck initialized')

    def get_all_layers(self):
        return [l for l in self.encoder.layers] + [l for l in self.decoder.layers]

    def fit(self, x_train, x_test=None, batch_size=128, epochs=1, lr=0.001, momentum=0.9, loss='mean_squared_error', callbacks=None):
        #self.model.compile(optimizer=SGD(lr=lr, momentum=momentum), loss=loss, metrics=metrics)
        self.model.compile(optimizer=Adam(lr=lr), loss=loss)
        return self.model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, x_test) if x_test is not None else None, callbacks=callbacks)

class CompositeModel(TrainableModel):
    def __init__(self, basemodel_params, split_layer, latent_dim):
        print('[info] comosite model, split_layer {}, latent_dim {}'.format(split_layer, latent_dim))
        self.basemodel_params = basemodel_params
        self.split_layer_name = split_layer
        self.latent_dim = latent_dim
        self.tasknet = EfficientnetClassifier(**basemodel_params)
        binput_shape = self.tasknet.base_model.get_layer(split_layer).output.shape[-3:]
        self.bottleneck = BottleNeck(latent_dim, binput_shape)

        self.model = insert_layers(self.tasknet.model, split_layer, self.bottleneck.get_all_layers(), position='after')

        self.input_shape = self.tasknet.input_shape
        self.intermediate_feature_extractor = tf.keras.models.Model(inputs=self.tasknet.model.input, outputs=self.tasknet.model.get_layer(split_layer).output)

        # define edge model
        x = self.tasknet.model.get_layer(split_layer).output
        x = self.bottleneck.encoder(x)
        self.edge_model = tf.keras.models.Model(inputs=self.tasknet.model.input, outputs=x)
        #print(self.model.summary())
        print('[info] composite model initialized')

    def freeze_layers(self, target='all'):
        assert target in ['all', 'feature_extractor', 'tasknet', 'bottleneck']
        print('[info] make {} of composite model untrainable'.format(target))
        if target == 'all':
            for layer in self.model.layers:
                layer.trainable = False
        elif target == 'feature_extractor':
            for layer in self.tasknet.feature_extractor.layers:
                layer.trainable = False
        elif target == 'tasknet':
            for layer in self.tasknet.model.layers:
                layer.trainable = False
        elif target == 'bottleneck':
            for layer in self.bottleneck.model.layers:
                layer.trainable = False

    def unfreeze_layers(self):
        print('[info] make all layers of composite model trainable')
        for layer in self.model.layers:
            layer.trainable = True

    def get_featuremap(self, dir, data_aug=False, batch_size=128):
        datagen = get_dir_generator(dir, self.input_shape[:2], train=data_aug, batch_size=batch_size)
        feature_map = []
        for num_batch, data in enumerate(datagen):
            x, y = data[0], data[1]
            output = self.intermediate_feature_extractor.predict(x)
            feature_map.extend(output.copy())
            if num_batch == len(datagen)-1: break
        return tf.convert_to_tensor(feature_map)

    def fit_bottleneck(self, dir, batch_size=128, epochs=1, lr=0.001, loss='mean_squared_error', callbacks=None, data_aug=False):
        #this is for task-agnostic training
        train_dir = os.path.join(dir, 'train') if os.path.exists(os.path.join(dir, 'train')) else dir
        test_dir = os.path.join(dir, 'val') if os.path.exists(os.path.join(dir, 'val')) else None

        train_fm = self.get_featuremap(train_dir, data_aug=data_aug, batch_size=batch_size)
        test_fm = self.get_featuremap(test_dir, batch_size=batch_size) if test_dir is not None else None

        return self.bottleneck.fit(train_fm, test_fm, batch_size=batch_size, epochs=epochs, lr=lr, loss=loss, callbacks=callbacks)

    def test(self, dir, batch_size=128):
        if isinstance(dir, str):
            test_dir = os.path.join(dir, 'val') if os.path.exists(os.path.join(dir, 'val')) else dir
            datagen = get_dir_generator(test_dir, self.input_shape[:2], train=False, batch_size=batch_size)
        else:
            x_test, y_test = dir[0], dir[1]
            datagen = get_generator(x_test, y_test, train=False, batch_size=batch_size)
        confidence, true_label, intermediate_featuremap, reconstructed = [], [], [], []
        for num_batch, data in enumerate(datagen):
            x, y = data[0], data[1]
            output = self.model.predict(x)
            confidence.extend(output.copy())
            true_label.extend(y)
            fm = self.intermediate_feature_extractor.predict(x)
            intermediate_featuremap.extend(fm.copy())
            recon = self.bottleneck.model.predict(fm)
            reconstructed.extend(recon.copy())
            if num_batch == len(datagen)-1: break
        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(true_label, confidence).numpy()
        print('loss: {}'.format(loss))
        true_labels = np.argmax(true_label, axis=1)
        preds = np.argmax(confidence, axis=1)
        acc = np.count_nonzero(true_labels==preds)/len(true_labels)
        print('acc: {}'.format(acc))
        cnf_mtx = sklearn.metrics.confusion_matrix(true_labels, preds)
        print('confusion matrix:')
        print(cnf_mtx.tolist())
        f1_score = sklearn.metrics.f1_score(true_labels, preds, average=None)
        print('f1 score: {}'.format(f1_score))
        mse = tf.keras.losses.MeanSquaredError()
        recon_loss = mse(intermediate_featuremap, reconstructed).numpy()
        print('reconstruction loss: {}'.format(recon_loss))
        return (acc, loss, recon_loss, cnf_mtx, f1_score)

######### Image data generator ##########
seed = 1024

datagen_params = dict(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False)

def get_dir_generator(dir, image_size, train=True, batch_size=128):
    datagen = ImageDataGenerator(**datagen_params) if train else ImageDataGenerator()
    return datagen.flow_from_directory(dir, target_size=image_size, batch_size=batch_size, shuffle=train, seed=seed)

def get_generator(x, y, train=True, batch_size=128):
    datagen = ImageDataGenerator(**datagen_params) if train else ImageDataGenerator()
    return datagen.flow(x, y, batch_size=batch_size, seed=seed)

######## COMMON #########

def convert_grayscale_to_rgb(arr):
    #convert grayscale image ndarray (n, x, y) to rgb tensor (n, x, y, 3)
    arr = np.expand_dims(arr, axis=3)
    ts = tf.convert_to_tensor(arr)
    return tf.image.grayscale_to_rgb(ts)

def insert_layers(model, name_of_layer_inserted, insert_layers, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if name_of_layer_inserted == layer.name:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            for new_layer in insert_layers:
                if not isinstance(new_layer, tf.keras.layers.InputLayer):
                    x = new_layer(x)

            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return tf.keras.models.Model(inputs=model.inputs, outputs=x)


def load_weights(model_cls, type, weight_path):
    if type is None: raise Exception('[err] weight path is specified but weight type not found')
    if type=='composite':
        assert isinstance(model_cls, CompositeModel)
        model_cls.model.load_weights(weight_path)
        print('[info] weights loaded on composite model')
    elif type=='bottleneck':
        assert isinstance(model_cls, CompositeModel)
        model_cls.bottleneck.model.load_weights(weight_path)
        print('[info] weights loaded on bottleneck of composite model')
    elif type=='classifier':
        if isinstance(model_cls, EfficientnetClassifier):
            model_cls.load_weights(weight_path)
            print('[info] weights loaded on efficientnet classifier')
        elif isinstance(model_cls, CompositeModel):
            model_cls.tasknet.load_weights(weight_path)
            print('[info] weights loaded on efficientnet classifier of composite model')
        else:
            raise Exception('[err] weight type mismatch. try to load {} on {} with {}'.format(weight_path, model_cls.__class__.__name__, type))
    else:
        assert isinstance(model_cls, EfficientnetRegressor)
        model_cls.load_weights(weight_path)
        print('[info] weights loaded on efficientnet regressor')

def load_all_weights(model_cls, args):
    if args.weight_path is None: return
    for type, weight_path in zip(args.weight_type, args.weight_path):
        load_weights(model_cls, type, weight_path)

######### training callbacks ##########
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

########### MAIN ###########
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['b0','b1','b2','b3','b4','b5','b6','b7'], default='b0', help='model name')
    parser.add_argument('--split_layer_name', type=str, help='name of layer which bottleneck layer is inserted after. specify only for composite model')
    parser.add_argument('--train_scheme', type=str, choices=['unperturbed', 'task_aware', 'task_agnostic', 'end_to_end'], default='unperturbed', help='training scheme')
    parser.add_argument('--weight_path', nargs='+', help='path to pretrain weight file')
    parser.add_argument('--weight_type', nargs='+', choices=['classifier','regressor','composite','bottleneck'], help='model type of pretrain weight file')
    parser.add_argument('--data_path', type=str, help='path to training and validation data')
    parser.add_argument('--save_path', type=str, help='path to trained weight file')
    parser.add_argument('--test', action='store_true', help='run test only')
    parser.add_argument('--data_augmentation', action='store_true', help='data augmentation will be enabled if specified')
    parser.add_argument('--lr', type=float, default=0.001, help='learnin rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--latent_dim', type=int, help='latent dim of composite model')
    parser.add_argument('--num_classes', type=int, help='number of classes of data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--set_callbacks', action='store_true')

    args = parser.parse_args()

    callbacks = [early_stopping, reduce_lr] if args.set_callbacks else []

    # composite model
    if args.split_layer_name is not None:
        efn_params = dict(model_name=args.model_name, num_classes=args.num_classes)
        composite_model = CompositeModel(efn_params, args.split_layer_name, args.latent_dim)
        load_all_weights(composite_model, args)
        #train model
        if not args.test:
            if args.train_scheme=='task_agnostic':
                composite_model.freeze_layers('tasknet')
                train_history = composite_model.fit_bottleneck(args.data_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, callbacks=callbacks)
            elif args.train_scheme=='task_aware':
                # task-aware training
                composite_model.freeze_layers('tasknet')
                train_history = composite_model.fit_dir(args.data_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, data_aug=args.data_augmentation, callbacks=callbacks)
            else:
                assert args.train_scheme=='end_to_end'
                train_history = composite_model.fit_dir(args.data_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, data_aug=args.data_augmentation, callbacks=callbacks)
            # save ONLY bottleneck
            if args.save_path is not None: composite_model.bottleneck.model.save_weights(args.save_path)
        #test model
        (acc, loss, recon_loss, cnf_mtx, f1_score) = composite_model.test(args.data_path, batch_size=args.batch_size)
        history = dict(accuracy=acc, loss=loss, recon_loss=recon_loss)
    # classification model
    elif args.num_classes is not None:
        efn_model = EfficientnetClassifier(num_classes=args.num_classes, model_name=args.model_name)
        load_all_weights(efn_model, args)
        if args.test:
            (acc, loss, cnf_mtx, f1_score) = efn_model.test(args.data_path, batch_size=args.batch_size)
            history = dict(accuracy=acc, loss=loss, recon_loss=None)
        else:
            history = efn_model.fit_dir(args.data_path, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, data_aug=args.data_augmentation, callbacks=callbacks)
            if args.save_path is not None: efn_model.save_weights(args.save_path)

    print('[info] result {}'.format(history))

    print('[info] done')
