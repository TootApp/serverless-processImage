import numpy as np
import os
import cv2
import base64
import math
import enchant
import pytesseract


def base64_to_image(b64_str, output='b64_orig_img.png'):
    """"Decodes a base64 str. To be used for generating images to be parsed
    :param b64_str: base64 string
    :param output: file name string"""

    img_data = base64.b64decode(b64_str)
    with open(output, 'wb') as f:
        f.write(img_data)


def image_to_base64(img_name):
    """"Encodes an image to base64 string format
    :param img_name: image file to be encoded"""
    with open(img_name, "rb") as f:
        data = f.read()
    return base64.b64encode(data)


def threshinv(image):
    """Applies a threshold filter, so that any value lighter than 160 will become 0, i.e. white.
    :param image: cv2-compatible image format
    :return: image after binary inversion"""

    _, thresh = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
    return thresh


# dilation over a thresholded image would cause any word to pixelate into one contour
def dilation(image):
    """Applies a dilate filter to enhance contours of characters
    :param image: cv2-compatible image format
    :return: image after dilate filtering"""

    a = math.sqrt(image.size)
    if a > 800:
        b = 7
    else:
        b = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b, b))
    result = cv2.dilate(image, kernel, iterations=1)
    return result


def clhe(image):
    """Histogram equalization of the image to improve constrast
    :param image: cbv2-compatible image format
    :return: image after clahe application"""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)
    return cl1


def total(img_to_proc):
    # thresholding and inversion of image for further processing.
    # handy for contour extraction
    img = img_to_proc
    temp = dilation(threshinv(clhe(img_to_proc)))
    _, contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    havg = 0
    wavg = 0
    wordsctr = 0
    total_area = 0
    thetaavg = 0

    for cnt in contours:
        a = img.size / cv2.contourArea(cnt)

        if 30 < a < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            wordsctr = wordsctr + 1
            rect = cv2.minAreaRect(cnt)
            # correction will be of no more than 45 degrees. Final orientation is assessed with reorient_image()
            if (rect[2]) < -45:
                thetaavg = thetaavg + (90 + rect[2]) * a
            elif (rect[2]) > 45:
                thetaavg = thetaavg - (90 + rect[2]) * a
            else:
                thetaavg = thetaavg + rect[2] * a
            total_area = total_area + a
            havg = havg + h
            wavg = wavg + w
    havg = havg / wordsctr
    wavg = wavg / wordsctr
    thetaavg = thetaavg / wordsctr / total_area
    rows, cols = img_to_proc.shape
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), thetaavg, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    width = int((rows * sin) + (cols * cos))
    height = int((rows * cos) + (cols * sin))
    M[0, 2] += (width / 2) - cols // 2
    M[1, 2] += (height / 2) - rows // 2
    output = cv2.warpAffine(img_to_proc, M, (width, height))
    cv2.fastNlMeansDenoising(output, output, h=3, templateWindowSize=7, searchWindowSize=21)
    return output, thetaavg, havg, wavg


def align_image(img_name, output='b64img.png'):
    """Reads the file path of the image to be rotated to an either horizontal or vertical orientation. It writes
    an image in gray scale, to be parsed later with tesseract. Function returns average height and width of
    bounding boxes to help estimate whether image is turned 90 degrees. To define the correct orientation, see
    reorient_image()
    :param img_name: image path
    :param output: output path
    :return: angle of rotation, average height and width of all the bounding boxes"""

    image = cv2.imread(img_name, 0)
    img1, angle, havg, wavg = total(image)
    cv2.imwrite(output, img1)
    return angle, havg, wavg


def reorient_image(theta, orientation, img_name='b64_orig_img.png'):
    """Reorients the original image in the direction the module believes to be the most accurate one.
    :param theta: angle in degrees for rotation, float
    :param orientation: integer identifying how many 90 degrees turns the image has to be rotated. See
                        sample_orientation()
    :param img_name: image to be rotated"""

    image = cv2.imread(img_name, cv2.IMREAD_COLOR)
    rows, cols, _ = image.shape
    angle = theta - orientation * 90
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), -angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    width = int((rows * sin) + (cols * cos))
    height = int((rows * cos) + (cols * sin))
    M[0, 2] += (width / 2) - cols // 2
    M[1, 2] += (height / 2) - rows // 2
    reor_img = cv2.warpAffine(image, M, (width, height))
    cv2.imwrite('b64_reor_img.png', reor_img)


def fetch_text(img):
    """Fetch text from image using tesseract. Uses bias for English.
    :param img: image to be analyzed """

    config = '-l eng --oem 1 --psm 3'
    return pytesseract.image_to_string(img, config=config)


def image_treating(img):
    """Applies additional filter to refine contrast.
    :param img: image to be filtered
    :return: filtered image in cv2-compatible format"""

    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, 25, -1, -1],
                       [-1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1]])

    return cv2.filter2D(img, -1, kernel)


def remove_temp():
    """Deletes all the files generated when using this module."""

    file_list = ['b64img.png', 'b64_orig_img.png', 'b64_reor_img.png']
    file_list.extend([f'sample_orientation{i}.png' for i in range(4)])

    for file in file_list:
        os.remove(file)


def sample_orientation(havg, wavg, img_name='b64img.png', dictionary=enchant.Dict("en_US")):
    """Inspect the four possible orientations for an aligned image. Text is extracted with fetch_text(). Images in
    the wrong direction will tend to result in random text, whereas for images correctly oriented will result in
    proper text. This function will count how many words are in a dictionary, and this will be used to decide
    the correct orientation. The code will also look into the bounding box dimensions, to be able to capture cases
    where the code mistakenly chooses a 90 degree rotated image
    :param havg: average height of bounding boxes
    :param wavg: average width of bounding boxes
    :param img_name: image to be analyzed
    :param dictionary: dictionary to be used for word counting
    :return: int number corresponding to the number of 90 degree rotations to be performed from the input image"""

    image = image_treating(cv2.imread(img_name, 0))
    cv2.imwrite(img_name, image)
    align_image(img_name)
    rows, cols = image.shape
    word_count = [0, 0, 0, 0]
    for i, angle in enumerate([0, 90, 180, 270]):
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        width = int((rows * sin) + (cols * cos))
        height = int((rows * cos) + (cols * sin))
        M[0, 2] += (width / 2) - cols // 2
        M[1, 2] += (height / 2) - rows // 2
        tmp_image = cv2.warpAffine(image, M, (width, height))
        cv2.imwrite(f'sample_orientation{i}.png', tmp_image)
        text = fetch_text(tmp_image)

        for word in text.split():
            if dictionary.check(word) and len(word) > 2:
                print(i, word)
                word_count[i] += 1

    max_word_count = word_count.index(max(word_count))
    """When havg is larger than wavg, most likely we are looking at a tilted reference, so we should exclude this one, 
    as well as the 180 degree rotation of it."""
    if havg > wavg and max_word_count == 0:
        word_count[0] = -1
        word_count[2] = -1  # This is the 180 degree rotation of the max count, so it should also be discarded

    elif havg < wavg and max_word_count % 2 is not 0:
        word_count[3] = -1
        word_count[1] = -1

    max_word_count = word_count.index(max(word_count))

    return max_word_count


def preprocess_image(image_name="b64_orig_img.png"):
    """Preprocesses image in colors to remove noise from it.
    :param image_name: image file (image converted from base64 str by default)"""

    img_in = cv2.imread(image_name)
    cv2.imwrite('b64img.png', cv2.fastNlMeansDenoisingColored(img_in, None, 10, 10, 7, 21))


def rotate_image(b64str):
    """Function that receives a base64 string of an image to be analyzed and rotated.
    :param b64str: base64 string of image
    :return: b64st of encoded rotated image"""

    base64_to_image(b64str)
    preprocess_image(image_name="b64_orig_img.png")
    angle, havg, wavg = align_image('b64img.png')
    orientation = sample_orientation(havg, wavg)
    reorient_image(angle, orientation, 'b64_orig_img.png')
    b64output = image_to_base64('b64_reor_img.png')
    remove_temp()

    return b64output

