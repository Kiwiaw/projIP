from unittest import TestCase
import cv2
from Localization import plate_detection

class TestPlateDetection(TestCase):
    def setUp(self):
        self.test_image_yellow_plate = "Yellow_Plate.jpeg"
        self.test_image_white_plate = "White_Plate.jpeg"
        self.test_image_black_plate = "Black_Plate.jpeg"

    def validate_plate_detection(self, image_path, plate_type):
        image = cv2.imread(image_path)
        self.assertIsNotNone(image, f"Test image '{plate_type}' could not be loaded")

        result = plate_detection(image)

        self.assertIsNotNone(result, f"Plate detection for {plate_type} plate returned None")
        self.assertTrue(result.size > 0, f"The cropped image for {plate_type} plate is empty")

        cv2.imshow(f"{plate_type.capitalize()} Plate - Detected Plate", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_plate_detection_yellow_plate(self):
        self.validate_plate_detection(self.test_image_yellow_plate, "yellow")

    def test_plate_detection_white_plate(self):
        self.validate_plate_detection(self.test_image_white_plate, "white")

    def test_plate_detection_black_plate(self):
        self.validate_plate_detection(self.test_image_black_plate, "black")

