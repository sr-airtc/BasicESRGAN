import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image
import numpy as np
from ESR_batch.model import GAN
import glob
def process_image(image):
    """Given an image, process it and return the array."""
    img = Image.open(image).convert("RGB")
    img = np.array(img)
    img = img.astype(np.float32)/127.5 - 1
    return np.expand_dims(img, axis=0)

def save_img(image, name):
    img = (image+1)*127.5
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save(os.path.join('sr', name))

model = GAN(3, 32)
model.load_weights('gan', 'checkpoints/0-gan.h5')
img_name = r'E:\Data\SR\DIV2K\DIV2K_test_LR_bicubic\X2\0901x2.png'
for name in glob.glob( r'E:\Data\SR\DIV2K\DIV2K_test_LR_bicubic\X4\*'):
    img = process_image(name)
    img = model.generator.model.predict(img)[0]
    save_img(img, os.path.basename(name))
