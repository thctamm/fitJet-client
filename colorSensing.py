import numpy as np
import cv2

class ColorSensing():

    def threshold(self, img):
        """
        Apply the threshold to the image to get only green parts
        :param img: RGB camera image
        :return: Masked image
        """
        # Define range of green color in HSV
        lower_green = np.array([40, 100, 50])
        upper_green = np.array([75, 255, 255])

        # Apply a blur
        img = cv2.medianBlur(img, 5)
        # Convert image to easier-to-work-with HSV format
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Threshold to get only green values
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        # Color space conversion back from HSV
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)

        return masked_img

    def bound_green_object(self, img):
        """
        Draw a bounding box around the largest green object in the scene
        :param img: RGB camera image
        :return: Image with bounding box
        """

        # Apply the threshold to get the green parts
        masked_img = self.threshold(img)
        x, y, w, h = (0, 0, 0, 0) 
        # Get contours
        img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Find the largest contour
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_contour = contours[max_index]
            max_contour_box = cv2.boundingRect(max_contour)
            for cont in contours:
                cont_box = cv2.boundingRect(cont)
                dist = self.get_bounds_distance(max_contour_box, cont_box)
                if dist > 0 and dist < 120:
                    max_contour_box = self.merge_bounds(max_contour_box, cont_box) 

            # Draw contours (for debugging)
            #cv2.drawContours(img, max_contour, -1, (0,255,0), 3)
            x, y, w, h = max_contour_box

            # Draw rectangle bounding box on image
            cv2.rectangle(masked_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        return (masked_img, (x, y, w, h))

    def find_bracelet(self, img):
        masked_img = self.threshold(img)
        return self.bound_green_object(masked_img)

    def get_bounds_distance(self, a, b):
        center_a = (a[0] + a[2]/2, a[1] + a[3]/2)
        center_b = (b[0] + b[2]/2, b[1] + b[3]/2)
        return ((abs(center_a[0] - center_b[0]))**2 + (abs(center_a[1] - center_b[1]))**2)**0.5

    def merge_bounds(self, a, b):
        x = a[0] if a[0] < b[0] else b[0]
        y = a[1] if a[1] < b[1]  else b[1]
        x_end = a[0] + a[2] if a[0] + a[2] > b[0] + b[2] else b[0] + b[2]
        y_end = a[1] + a[3] if a[1] + a[3] > b[1] + b[3] else b[1] + b[3]

        w = x_end - x
        h = y_end - y

        return (x, y, w, h)

