

from tensorflow.keras.utils import image_dataset_from_directory
import os

def image_segmentation_generator(data_dir, batch_size, target_size=(256, 256)):
    mask_gen = image_dataset_from_directory(
        r"E:\deployment_spam\UNet\data_col\New folder", # Assuming separate image and mask subdirectories
        # classes=None,  # No class labels needed for segmentation
        image_size=target_size,
        batch_size=batch_size
    )
    # image_gen = image_dataset_from_directory(  # Assuming masks in a subdirectory named "masks"
    #     r"E:\deployment_spam\UNet\data_col\New folder (2)",
    #     # classes=None,
    #     image_size=target_size,
    #     batch_size=batch_size,
    # )

    # Zip image and mask generators for paired data
    # image_mask_pair = zip(image_gen, mask_gen)

    for images, masks in mask_gen:
        # Preprocess images and masks here (e.g., normalize)
        yield images
    # return mask_gen,image_gen
# Example usage in model.fit:

