import segmentation_models as sm
import matplotlib.pyplot as plt
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import numpy as np
import cv2 
from segmentation_models.utils import set_trainable
from data_pre import image_segmentation_generator
BACKBONE = 'resnet152'
preprocess_input = sm.get_preprocessing(BACKBONE)

# model = sm.Unet(BACKBONE, encoder_weights='imagenet')


train_datax = image_segmentation_generator(r"E:\deployment_spam\UNet\data_col\New folder",batch_size=1)
train_datay = image_segmentation_generator(r"E:\deployment_spam\UNet\data_col\New folder (2)",batch_size=1)
train_data = zip(train_datax,train_datay)
# model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
# model.fit(train_data,epochs=10)
# # print(train_datax.class_names)
# for i,a in train_data:
#     print(i,a)
#     # break
# model.load_weights('./model1.h5')

# model = sm.Unet(backbone_name='resnet34', encoder_weights='imagenet', encoder_freeze=True)
# model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
# model.fit(train_data, epochs=2)

# for images, labels in train_data:
#   for i in range(1):
#     # ax = plt.subplot(6, 6, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     # plt.title(class_names[np.argmax(labels[i])])
#     plt.axis("off")
#     break

# print(s.shape)
# for i in range(1):
#     plt.imshow(sdf[i].astype("uint8"))
# plt.show()