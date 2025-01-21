from unittest import TestCase
import cv2
from Localization import plate_detection
from main import addDutchDashes


# class TestPlateDetection(TestCase):
#     def setUp(self):
#         self.test_image_yellow_plate = "Yellow_Plate.jpeg"
#         self.test_image_white_plate = "White_Plate.jpeg"
#         self.test_image_black_plate = "Black_Plate.jpeg"
#
#     def validate_plate_detection(self, image_path, plate_type):
#         image = cv2.imread(image_path)
#         self.assertIsNotNone(image, f"Test image '{plate_type}' could not be loaded")
#
#         result = plate_detection(image)
#
#         self.assertIsNotNone(result, f"Plate detection for {plate_type} plate returned None")
#         self.assertTrue(result.size > 0, f"The cropped image for {plate_type} plate is empty")
#
#         cv2.imshow(f"{plate_type.capitalize()} Plate - Detected Plate", result)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     def test_plate_detection_yellow_plate(self):
#         self.validate_plate_detection(self.test_image_yellow_plate, "yellow")
#
#     def test_plate_detection_white_plate(self):
#         self.validate_plate_detection(self.test_image_white_plate, "white")
#
#     def test_plate_detection_black_plate(self):
#         self.validate_plate_detection(self.test_image_black_plate, "black")

# platesGT = ["XX-99-99", "99-99-XX", "99-XX-99", "XX-99-XX", "XX-XX-99", "99-XX-XX", "99-XXX-9", "9-XXX-99", "XX-999-X", "X-999-XX", "XXX-99-X", "XXX-99-X", "X-99-XXX", "9-XX-999", "999-XX-9"]
# plates = ["XX9999", "9999XX", "99XX99", "XX99XX", "XXXX99", "99XXXX", "99XXX9", "9XXX99", "XX999X", "X999XX", "XXX99X", "XXX99X", "X99XXX", "9XX999", "999XX9"]
# for i, plate in enumerate(plates):
#     try:
#         assert(addDutchDashes(plate) == platesGT[i])
#     except AssertionError:
#         print(f"Expected format: {platesGT[i]}")
#         print(f"Result after addDutchDashes: {addDutchDashes(plate)}\n")