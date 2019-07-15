import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

from ESR_batch.model import GAN
from ESR_batch.data import  dataGenertor
import numpy as np

epochs = 100
hr_batch_size = 1
lr_batch_size = 1
step_epoch = 1
crop_w = 128
crop_h = 256

if __name__ == '__main__':
    train_lr = dataGenertor(
        r"E:\Data\SR\DIV2K\DIV2K_train_LR_bicubic\X4",
        batch_size=lr_batch_size,
        crop_w=crop_w,
        crop_h=crop_h
    )
    val_lr = dataGenertor(
        r'E:\Data\SR\DIV2K\DIV2K_test_LR_bicubic\X4',
        batch_size=lr_batch_size,
        crop_w=crop_w,
        crop_h=crop_h
    )

    train_hr = dataGenertor(
        r'E:\Data\SR\DIV2K\DIV2K_train_HR',
        batch_size=hr_batch_size,
        expansion=4,
        crop_w=crop_w,
        crop_h=crop_h
    )
    model = GAN(3, 32, discriminator_name='MobileNetV2')
    # model.load_weight('gan_g', r'E:\Learn\XSSR\ESR\checkpoints\2019-07-14\0.000-1.000-ESRGAN.h5')
    for epoch in range(epochs):
        discriminator_loss = np.zeros([2])
        gan_loss = np.zeros([2])
        batch = 0
        for lr_imgs, hr_imgs in zip(train_lr, train_hr):
            sr_imgs = model.generator.predict(lr_imgs)

            batch_imgs_d = np.concatenate([sr_imgs, hr_imgs], axis=0)
            batch_labels_d = np.array([0]*lr_batch_size + [1]*hr_batch_size)
            model.set_preparation('discriminator')
            # print('training discriminator...')
            batch_d_loss = model.discriminator.train_on_batch(batch_imgs_d, batch_labels_d)

            lr_labels = np.ones((lr_batch_size))
            model.set_preparation('gan')
            # print('training gan...')
            batch_g_loss = model.gan.train_on_batch(lr_imgs, lr_labels)

            if batch%2 == 0:
                print('[%d batch] discriminator loss %.3f, acc:%.3f; gan loss:%.3f, acc: %.3f' %
              (epoch, batch_d_loss[0], batch_d_loss[1], batch_g_loss[0], batch_g_loss[1]))

            batch += 1
            discriminator_loss += batch_d_loss
            gan_loss += batch_g_loss

        print('[%d iter] discriminator loss %.3f, acc:%.3f; gan loss:%.3f, acc: %.3f' %
              (epoch, discriminator_loss[0], discriminator_loss[1], gan_loss[0], gan_loss[1]))





