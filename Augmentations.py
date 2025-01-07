import albumentations as A
import cv2
import numpy as np
import os

import Recognize


def augmentationImages(original_image_name,image, label, output_dir, label_file, num_augments=10):

    transform = A.Compose([
        A.Rotate(limit=(-20, 20), p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ToGray(p=0.5),
    ])

    os.makedirs(output_dir, exist_ok=True)
    padded_image = add_border(image, border_size=20)


    with open(label_file, 'a') as label_out:
        for i in range(num_augments):
            augmented = transform(image=padded_image)
            augmented_image = augmented['image']

            cleaned_image = clean_image(augmented_image)

            output_filename = f"{original_image_name}_aug_{i}.bmp"  # Save as BMP
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cleaned_image)


            label_out.write(f"{output_filename} {label}\n")

    print(f"Created to {output_dir} and labels added to:  {label_file}.")


def add_border(image, border_size=20):

    return cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )


def clean_image(image):

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.erode(image, kernel, iterations=1)
    return cleaned



if __name__ == "__main__":

    for i in range(1, 28):
        xCurrent, yCurrent = Recognize.getPhotoPath(i)
        input_image = cv2.imread(xCurrent)

        label = yCurrent
        original_image_base_name = os.path.splitext(os.path.basename(xCurrent))[0]

        original_image_name = f"{original_image_base_name}_{i}"
        output_directory = "dataset/AugmentedImages"
        label_file_path = os.path.join(output_directory, "labels")


        augmentationImages(
            original_image_name,
            input_image,
            label,
            output_directory,
            label_file_path,
            num_augments=10
        )