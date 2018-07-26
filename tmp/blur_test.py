import cv2
import numpy as np


def blur(image):
    # detect edge
    edge = cv2.Canny(image, np.random.randint(75, 125), np.random.randint(175, 225))

    # dilate edge
    d_size = np.random.randint(2, 7)
    dilated_edge = cv2.dilate(edge, np.ones((d_size, d_size), np.uint8), iterations=1)

    # gray to bgr
    dilated_edge = cv2.cvtColor(dilated_edge, cv2.COLOR_GRAY2BGR)

    # concat and blur
    b_size = np.random.randint(1, 5) * 2 + 1
    gaussian_smoothing_image = cv2.GaussianBlur(image, (b_size, b_size), 0)
    cv2.imwrite('gaussian.png', gaussian_smoothing_image)

    merged = np.where(dilated_edge > 128, gaussian_smoothing_image, image)
    return merged


img = cv2.imread('/Users/la4la/PycharmProjects/cartoongan/15425_img471_bg_eki_hiru_0734.png')
blured = blur(img)
cv2.imwrite('blured.png', blured)