from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import SGD, Adam
from functools import partial
import os

class ResidualDenseBlock_5c:
    def __init__(self, nf=64, gc=32, bias=True):
        self.model = self.get_model(nf, gc, bias)

    def get_model(self, nf, gc, bias):
        input_x = Input(shape=(None, None, nf))
        concat = [input_x]
        x = input_x
        for i in range(4):
            x = Conv2D(gc, (3, 3), (1, 1), padding="same", use_bias=bias, activation=LeakyReLU(0.2))(x)
            concat.append(x)
            x = Concatenate(axis=-1)(concat)
        x = Conv2D(nf, (3, 3), (1, 1), padding="same", use_bias=bias)(x)
        res_x = Lambda(lambda x: 0.1 * x)(input_x)
        output = Add()([res_x, x])
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class RRDB:
    def __init__(self, nf, gc=32):
        self.model = self.get_model(nf, gc)

    def get_model(self, nf, gc):
        input_x = Input(shape=(None, None, nf))
        RDB1 = ResidualDenseBlock_5c(nf, gc)
        RDB2 = ResidualDenseBlock_5c(nf, gc)
        RDB3 = ResidualDenseBlock_5c(nf, gc)
        
        x = RDB1(input_x)
        x = RDB2(x)
        x = RDB3(x)

        res_x = Lambda(lambda x: 0.1 * x)(input_x)
        output = Add()([res_x, x])
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)


class RRDBNet:
    def __init__( self, out_nc, nf, nb=23, gc=32):
        self.model = self.get_model(out_nc, nf, nb, gc)

    def get_model(self, out_nc, nf, nb, gc):
        RRBD_block_f = partial(RRDB, nf=nf, gc=gc)
        input_x = Input(shape=(None, None, 3))
        fea = Conv2D(nf, (3, 3), (1, 1), padding="same")(input_x)
        x = fea
        for i in range(nb):
            x = RRBD_block_f()(x)
        x = Add()([x, fea])

        x = UpSampling2D((2, 2))(x)
        x = LeakyReLU(0.2)(x)
        x = UpSampling2D((2, 2))(x)
        x = LeakyReLU(0.2)(x)
        output = Conv2D(out_nc, (3, 3), padding="same")(x)
        model = Model(input_x, output)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)

    def predict(self, imgs):
        return self.model.predict(imgs)

    def predict_on_batch(self, imgs):
        return self.model.predict(imgs)

    def train_on_batch(self, x, y):
        self.model.train_on_batch(x, y)

class ImageNet:
    def __init__(self, model_name, weights="imagenet"):
        self.model = self.get_model(model_name, weights)

    def get_model(self, model_name, weights):
        if model_name == 'InceptionV3':
            from tensorflow.python.keras.applications.inception_v3 import InceptionV3
            base_model = InceptionV3(weights=weights, include_top=False)
        elif model_name == 'NASNetLarge':
            from tensorflow.python.keras.applications.nasnet import NASNetLarge
            base_model = NASNetLarge(weights=weights, include_top=False)
        elif model_name == 'DenseNet201':
            from tensorflow.python.keras.applications.densenet import DenseNet201
            base_model = DenseNet201(weights=weights, include_top=False)
        elif model_name == 'Xception':
            from tensorflow.python.keras.applications.xception import Xception
            base_model = Xception(weights=weights, include_top=False)
        elif model_name == 'VGG16':
            from tensorflow.python.keras.applications.vgg16 import VGG16
            base_model = VGG16(weights=weights, include_top=False)
        elif model_name == 'VGG19':
            from tensorflow.python.keras.applications.vgg19 import VGG19
            base_model = VGG19(weights=weights, include_top=False)
        elif model_name == 'NASNetMobile':
            from tensorflow.python.keras.applications.nasnet import NASNetMobile
            base_model = NASNetMobile(weights=weights, include_top=False)
        elif model_name == 'MobileNetV2':
            from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
            base_model = MobileNetV2(weights=weights, include_top=False)
        elif model_name == 'ResNet50':
            from tensorflow.python.keras.applications.resnet50 import ResNet50
            base_model = ResNet50(weights=weights, include_top=False)
        elif model_name == 'InceptionResNetV2':
            from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
            base_model = InceptionResNetV2(weights=weights, include_top=False, )
        else:
            raise KeyError('Unknown network.')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def __call__(self, x, *args, **kwargs):
        return self.model(x)

    def predict(self, imgs):
        return self.model.predict(imgs)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)




class GAN:
    def __init__(self, out_nc, nf=64, nb=23, gc=32, discriminator_name='VGG16', discriminator_weights="imagenet"):
        self.generator = RRDBNet(out_nc, nf, nb=nb, gc=gc)
        self.discriminator = ImageNet(discriminator_name, discriminator_weights)
        self.discriminator.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # print('gan_d')
        self.gan = self.get_GAN_model()
        self.gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.gan.summary()
        # print('gan_d')
        # self.gan_d = self.get_GAN_D_model()
        # self.gan_d.summary()
        self.fake_imgs = []

    def get_GAN_model(self):
        input_x = Input(shape=(None, None, 3))
        x = self.generator(input_x)
        output = self.discriminator(x)
        model = Model(input_x, output)
        return model

    def train_generator(self, epochs, train, val):
        # if not hasattr(self, "callback"):
        #     self.callback = self.get_callback()
        self.discriminator.model.trainable = False
        # self.gan.layers[1].trainable = True
        self.gan.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        # self.gan.summary()
        loss = self.gan.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            validation_data=val,
            validation_steps=len(val),
            epochs=epochs,
            workers=8,
            use_multiprocessing=True,
            max_queue_size=100,
        )

        # for layer in self.discriminator.model.layers[-60:]:
        #     layer.trainable = True
        self.discriminator.model.trainable = True

        return loss

    def train_discriminator(self, epochs, train):
        # if not hasattr(self, "callback"):
        #     self.callback = self.get_callback()
        self.discriminator.model.trainable = True
        self.discriminator.model.compile(optimizer=SGD(1e-4, 0.9), loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.model.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            epochs=epochs,
            # workers=8,
            # use_multiprocessing=True,
            # callbacks=self.callback
        )

    def set_preparation(self, model_type):
        if model_type not in ['gan', 'discriminator']:
            raise ValueError('model_type must be in [gan, discriminator]!!!')
        if model_type == 'gan':
            self.discriminator.model.trainable = False
            self.gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # self.gan.summary()
        else:
            self.discriminator.model.trainable = True
            self.discriminator.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # self.discriminator.model.summary()

    def get_callback(self):
        base_path = 'checkpoints'
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        import datetime
        today = datetime.datetime.today()
        name = '%s' % today
        self.name = name.split(' ')[0]
        self.name = os.path.join(base_path, self.name)
        if not os.path.exists(self.name):
            os.mkdir(self.name)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.name, '{val_loss:.3f}-{val_acc:.3f}-ESRGAN.h5'),
            verbose=1,
            save_best_only=True,
            monitor='val_loss'
        )
        return [checkpointer]


    def load_weights(self, model_type, model_weights):
        if model_type not in ['gan', 'discriminator']:
            raise ValueError('model_type must be in [gan, discriminator]!!!')
        print('加载%s模型文件%s'%(model_type, model_weights))
        if model_type == 'gan':
            eval("self."+model_type).load_weights(model_weights)
        else:
            eval("self." + model_type).model.load_weights(model_weights)

    def save_weights(self, model_type, epoch=-1):
        if model_type not in ['gan', 'discriminator']:
            raise ValueError('model_type must be in [gan, discriminator]!!!')
        if model_type == 'gan':
            print('保存gan网络...')
            eval("self."+model_type).save_weights('checkpoints/%d-gan.h5' % epoch)
        else:
            print('保存discriminator网络...')
            eval("self." + model_type).model.save_weights('checkpoints/%d-discriminator.h5' % epoch)
