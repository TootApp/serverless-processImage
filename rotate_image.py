import numpy as np
import os
import cv2
import base64
import math
import enchant
import boto3


def base64_to_image_old(b64_str, output='b64_orig_img.png'):
    img_data = base64.b64decode(b64_str)
    with open(output, 'wb') as f:
        f.write(img_data)


def base64_to_image(b64_str):
    """"Decodes a base64 str. To be used for generating images to be parsed
    :param b64_str: base64 string"""
    img = base64.b64decode(b64_str)
    img_array = np.fromstring(img, np.uint8)
    return cv2.imdecode(img_array, 1)


def image_to_base64_old(img_name):
    """"Encodes an image to base64 string format
    :param img_name: image file to be encoded"""
    with open(img_name, "rb") as f:
        data = f.read()
    return base64.b64encode(data)


def image_to_base64(image):
    """"Encodes an image to base64 string format
    :param image: image object to be encoded"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer)


def image_to_bytes(image):
    """Converts an image object to a byte array
    :param image: image object"""
    return cv2.imencode('.png', image)[1].tostring()


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


def straighten_image(img_to_proc, areas, angles):
    # thresholding and inversion of image for further processing.
    # handy for contour extraction

    total_area = 0
    thetaavg = 0

    for i, area in enumerate(areas):
            # correction will be of no more than 45 degrees. Final orientation is assessed with reorient_image()
        if (angles[i]) < -45:
            thetaavg = thetaavg + (90 + angles[i]) * area
        elif (angles[i]) > 45:
            thetaavg = thetaavg - (90 + angles[i]) * area
        else:
            thetaavg = thetaavg + angles[i] * area
        total_area = total_area + area

    thetaavg = thetaavg / len(angles) / total_area
    rows, cols, _ = img_to_proc.shape
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), thetaavg, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    width = int((rows * sin) + (cols * cos))
    height = int((rows * cos) + (cols * sin))
    M[0, 2] += (width / 2) - cols // 2
    M[1, 2] += (height / 2) - rows // 2
    output = cv2.warpAffine(img_to_proc, M, (width, height))
    #cv2.fastNlMeansDenoising(output, output, h=3, templateWindowSize=7, searchWindowSize=21)
    return output, thetaavg


def align_image(image):
    """Reads the file path of the image to be rotated to an either horizontal or vertical orientation. It writes
    an image in gray scale, to be parsed later with tesseract. Function returns average height and width of
    bounding boxes to help estimate whether image is turned 90 degrees. To define the correct orientation, see
    reorient_image()
    :param image: image object"""

#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, areas, angles = fetch_text(image_to_bytes(image))
    output, theta = straighten_image(image, areas, angles)
    return output, theta


def reorient_image(image, theta, orientation):
    """Reorients the original image in the direction the module believes to be the most accurate one.
    :param theta: angle in degrees for rotation, float
    :param orientation: integer identifying how many 90 degrees turns the image has to be rotated. See
                        sample_orientation()
    :param image: image to be rotated"""

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

    return reor_img


def fetch_text(img_bytes): # img_name):
    """Fetch text from image using AWS rekognition.
    :param img_bytes: byte array encoding an image
    :return: json of response from rekognition detect_text()"""

    rekognition = boto3.client('rekognition')
    response = rekognition.detect_text(
            Image={
                'Bytes': img_bytes,
                }
        )

    text = []
    widths = []
    heights = []
    areas = []
    angles = []

    for box in response['TextDetections']:
        if box['Confidence'] > 90:
            text.append(box['DetectedText'])
            widths.append(box['Geometry']['BoundingBox']['Width'])
            heights.append(box['Geometry']['BoundingBox']['Height'])
            areas.append(widths[-1] * heights[-1])
            pointA = box['Geometry']['Polygon'][0]
            pointB = box['Geometry']['Polygon'][1]
            pointC = box['Geometry']['Polygon'][2]
            AB = math.sqrt(pow(pointB['X'] - pointA['X'], 2) + pow(pointB['Y'] - pointA['Y'], 2))
            BC = math.sqrt(pow(pointB['X'] - pointC['X'], 2) + pow(pointB['Y'] - pointC['Y'], 2))

            if np.abs(heights[-1]) <= np.abs(widths[-1]):
                # Box larger in width
                if AB <= BC:
                    # BC is our width
                    angles.append(180 * math.atan2(pointB['Y'] - pointC['Y'], pointB['X'] - pointC['X'])/np.pi)
                else:
                    # AB is our width
                    angles.append(180 * math.atan2(pointB['Y'] - pointA['Y'], pointB['X'] - pointA['X'])/np.pi)

            else:
                """Tall box. It's possible to get single numbers, such as 1, which is tall and thin. Need to account 
                for this"""
                if len(text[-1]) < 2:
                    # Single tall and thin character case
                    if AB > BC:
                        # CD is our width
                        angles.append(180 * math.atan2(pointB['Y'] - pointC['Y'], pointB['X'] - pointC['X']) / np.pi)
                    else:
                        # AB is our width
                        angles.append(180 * math.atan2(pointB['Y'] - pointA['Y'], pointB['X'] - pointA['X']) / np.pi)

                else:
                    if AB <= BC:
                        # CD is our height
                        angles.append(180 * math.atan2(pointB['Y'] - pointC['Y'], pointB['X'] - pointC['X']) / np.pi)
                    else:
                        # AB is our height
                        angles.append(180 * math.atan2(pointB['Y'] - pointA['Y'], pointB['X'] - pointA['X']) / np.pi)

    return text, areas, angles


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


def sample_orientation(image, dictionary=enchant.Dict("en_US")):
    """Inspect the four possible orientations for an aligned image. Text is extracted with fetch_text(). Images in
    the wrong direction will tend to result in random text, whereas for images correctly oriented will result in
    proper text. This function will count how many words are in a dictionary, and this will be used to decide
    the correct orientation. The code will also look into the bounding box dimensions, to be able to capture cases
    where the code mistakenly chooses a 90 degree rotated image
    :param img_gray: image object in gray scale
    :param havg: average height of bounding boxes
    :param wavg: average width of bounding boxes
    :param dictionary: dictionary to be used for word counting
    :return: int number corresponding to the number of 90 degree rotations to be performed from the input image"""

    # img_gray = image_treating(img_gray)
    rows, cols, _ = image.shape
    word_count = [0, 0, 0, 0]
    for orientation, angle in enumerate([0, 90, 180, 270]):
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        width = int((rows * sin) + (cols * cos))
        height = int((rows * cos) + (cols * sin))
        M[0, 2] += (width / 2) - cols // 2
        M[1, 2] += (height / 2) - rows // 2
        tmp_image = cv2.warpAffine(image, M, (width, height))
        img_bytes = image_to_bytes(tmp_image)
        text, areas, angles = fetch_text(img_bytes)

        if len(text) == 0:
            continue

        """Cutting out images with too much vertical text from the beginning"""
        if np.abs(np.average(angles, weights=areas)) > 45:
            word_count[orientation] = -1
            continue

        else:
            for i, word in enumerate(text):
                if dictionary.check(word) and np.abs(angles[i]) < 45 and word.isalnum():
                    word_count[orientation] += 1

        """Particularly for equations as text, it can be tricky. We thus need to rely on counting of boxes only"""

        if max(word_count) == 0:
            for i, word in enumerate(text):
                if np.abs(angles[i]) < 45:
                    word_count[orientation] += 1

    return word_count.index(max(word_count))


def preprocess_image(image):
    """Preprocesses image in colors to remove noise from it.
    :param image object
    :return: image object after color filtering"""

    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def rotate_image(b64str):
    """Function that receives a base64 string of an image to be analyzed and rotated.
    :param b64str: base64 string of image
    :return: b64st of encoded rotated image"""

    image = base64_to_image(b64str)
    image = preprocess_image(image)
    image, angle = align_image(image)
    orientation = sample_orientation(image)
    output = reorient_image(image, angle, orientation)
    b64output = image_to_base64(output)

    return b64output
