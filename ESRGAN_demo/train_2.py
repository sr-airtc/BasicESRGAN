# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# KTF.set_session(sess)

from ESR_batch.model import GAN, SGD, Adam
from ESR_batch.data import dataDiscriminator, dataGenertor
import numpy as np

epochs = 100
d_batch_size = 2
g_batch_size = 1
step_epoch = 1
crop_w = 128
crop_h = 128

if __name__ == '__main__':
    model = GAN(3, 32)

    # model.load_weights('discriminator', r'E:\Learn\XSSR\ESR_batch\checkpoints\0-discriminator.h5')
    # for layer in model.discriminator.model.layers[l//2:]:
    #     layer.trainable = False
    model.gan.summary()
    model.load_weights('gan', 'checkpoints/0-gan.h5')
    model.load_weights('discriminator', 'checkpoints/0-discriminator.h5')
    # model.load_weights('gan', r'E:\Learn\XSSR\ESR\checkpoints\2019-07-14\0.000-0.000-ESRGAN.h5')
    train_g = dataGenertor(r"E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4", batch_size=g_batch_size, crop_w=crop_w, crop_h=crop_h, return_labels=True)
    val_g = dataGenertor(r'E:\Data\SR\DIV2K\DIV2K_test_LR_bicubic\X4', batch_size=g_batch_size, crop_w=crop_w, crop_h=crop_h, return_labels=True)

    train_d = dataDiscriminator(r'E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4',
                                r'E:\Data\SR\DIV2K\DIV2K_train_HR',
                                model=model.generator,
                                batch_size=d_batch_size
                                )
    # model.discriminator.model.compile(optimizer=SGD(1e-2, 0.9), loss='binary_crossentropy', metrics=['accuracy'])
    # for batch, (x, y) in enumerate(train_d):
    #     loss = model.discriminator.train_on_batch(x, y)
    #     if batch%10 == 0:
    #         print(loss)

    for epoch in range(epochs):
        low_loss_batch = 0
        model.discriminator.model.compile(optimizer=SGD(1e-5, 0.9, 0.01), loss='binary_crossentropy', metrics=['accuracy'])
        total_loss = np.zeros((2))
        for batch, (x, y) in enumerate(train_d):
            loss = model.discriminator.train_on_batch(x, y)
            total_loss += loss
            if total_loss[1]/(batch+1) >= 0.99:
                if low_loss_batch > 50:
                    break
                low_loss_batch += 1
            else:
                low_loss_batch = 0
            if batch % 100 == 99:
                print('[%d epoch %d batch] discriminator loss %.3f, acc:%.3f' %
                      (epoch, batch, total_loss[0]/(batch+1), total_loss[1]/(batch+1)))
                if batch>=4000 and total_loss[1]/(batch+1)>=0.95:
                    break
        model.save_weights('discriminator', epoch)

        model.train_generator(step_epoch, train_g, val_g)
        # for _ in range(10):
        #     loss = model.train_generator(step_epoch, train_g, val_g)
        #     if loss[0] <= 1e-2:
        #         break
        model.save_weights('gan', epoch)

