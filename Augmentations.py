import albumentations as A
import cv2
import numpy as np
import os

import Recognize
from albumentations.core.transforms_interface import ImageOnlyTransform

class AddNoiseAroundCharacters(ImageOnlyTransform):
    def __init__(self, noise_intensity=10, noise_radius=5, p=1.0):
        super().__init__(always_apply=False, p=p)
        self.noise_intensity = noise_intensity
        self.noise_radius = noise_radius

    def apply(self, image, **params):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        noise_mask = cv2.dilate((image > 128).astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                          (self.noise_radius, self.noise_radius)))

        noise = np.random.normal(0, self.noise_intensity, image.shape).astype(np.uint8)
        noise = cv2.bitwise_and(noise, noise, mask=noise_mask)

        noisy_image = cv2.add(image, noise)
        return noisy_image

    def get_transform_init_args_names(self):
        return ("noise_intensity", "noise_radius")




def ContentCentered(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours((image > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped_content = image[y:y + h, x:x + w]


        size = max(cropped_content.shape)
        padded = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - cropped_content.shape[0]) // 2
        x_offset = (size - cropped_content.shape[1]) // 2
        padded[y_offset:y_offset + cropped_content.shape[0],
        x_offset:x_offset + cropped_content.shape[1]] = cropped_content

        return cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)


def augmentationImages(original_image_name,image, label, output_dir, label_file, num_augments=10):

    transform = A.Compose([
        A.Rotate(limit=(-15, 15), p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Perspective(scale=(0.03, 0.1), keep_size=True, pad_val=0, p=0.4),
        AddNoiseAroundCharacters(noise_intensity=10, noise_radius=3, p=0.3),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.4),
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        # A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),

    ])

    os.makedirs(output_dir, exist_ok=True)
    padded_image = add_border(image, border_size=20)


    with open(label_file, 'a') as label_out:
        for i in range(num_augments):
            augmented = transform(image=padded_image)
            augmented_image = augmented['image']
            # center lol
            centered_image = ContentCentered(augmented_image)

            cleaned_image = clean_image(centered_image)

            output_filename = f"{original_image_name}_aug_{i}.bmp"
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
        #so there is less incorrect classsifications as 1's
        num_augments = 15 if i ==6 else 0 if i ==19 else 15
        augmentationImages(
            original_image_name,
            input_image,
            label,
            output_directory,
            label_file_path,
            num_augments=num_augments
        )