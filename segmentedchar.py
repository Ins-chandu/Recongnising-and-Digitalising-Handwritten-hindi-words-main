import cv2
import numpy as np
from scipy import ndimage
import math
import argparse
import os
import tensorflow as tf
import all_functions_used as helpers

# mapping ={0:u'ठ',1:u'ड',2:u'त',3:u'थ',4:u'द',5:u'क',6:u'न',7:u'प',8:u'फ',9:u'म',10:u'य',11:u'र',12:u'व',13:u'स',14:u'क्ष',15:u'त्र',16:u'ज्ञ',17:u'घ',18:u'च',19:u'छ',20:u'ज'}

# modelpath=".\\Models\\21classesmodel.h5"

mapping = {
    0: "ञ",
    1: "ट",
    2: "ठ",
    3: "ठ",
    4: "ढ",
    5: "ण",
    6: "त",
    7: "थ",
    8: "द",
    9: "ध",
    10: "क",
    11: "न",
    12: "प",
    13: "फ",
    14: "ब",
    15: "भ",
    16: "म",
    17: "य",
    18: "र",
    19: "ल",
    20: "व",
    21: "ख",
    22: "श",
    23: "ष",
    24: "स",
    25: "ह",
    26: "क्ष",
    27: "त्र",
    28: "ज्ञ",
    29: "ग",
    30: "घ",
    31: "ङ",
    32: "च",
    33: "छ",
    34: "ज",
    35: "झ",
    36: "०",
    37: "१",
    38: "२",
    39: "३",
    40: "४",
    41: "५",
    42: "६",
    43: "७",
    44: "८",
    45: "९",
}

modelpath = ".\\Models\\46classesmodel.h5"

# Load model in inference mode to avoid optimizer deserialization issues
# from legacy training configs stored in older .h5 files.
model = tf.keras.models.load_model(modelpath, compile=False)


def predict(imagepath, output_dir):
    gray_img = helpers.load_image(imagepath)
    print(gray_img.shape)

    adap_threshold_img = helpers.remove_noise_and_preprocess(gray_img)
    adap_threshold_img_path = os.path.join(
        "./segmented_characters/", f"adap_thresh_img.png"
    )
    cv2.imwrite(adap_threshold_img_path, adap_threshold_img)

    otsu_threshold_img = helpers.preprocess(gray_img)
    otsu_threshold_img_path = os.path.join(
        "./segmented_characters/", f"otsu_thresh_img.png"
    )
    cv2.imwrite(otsu_threshold_img_path, otsu_threshold_img)

    # Perform bitwise and on both adaptive threshold image and otsu threhold image
    # and again store in otsu threshold image
    for i in range(otsu_threshold_img.shape[0]):
        for j in range(otsu_threshold_img.shape[1]):
            if adap_threshold_img[i][j] == 255 and otsu_threshold_img[i][j] == 255:
                continue
            else:
                otsu_threshold_img[i][j] = 0

    cv2.imshow("processed_image", otsu_threshold_img)
    cv2.waitKey(1000)
    processed_img_path = os.path.join("./segmented_characters/", f"proc_img.png")
    cv2.imwrite(processed_img_path, otsu_threshold_img)

    # returns coordinates of header line and rotate the image horizontally
    x1, x2, y1, y2 = helpers.houghtransform(otsu_threshold_img)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    rotated_img = ndimage.rotate(otsu_threshold_img, angle)

    rotated_img = helpers.word_segmentation(rotated_img)

    cv2.imshow("Rotated_word_image", rotated_img)
    cv2.waitKey(1000)
    rotated_img_path = os.path.join("./segmented_characters/", f"rotated_img.png")
    cv2.imwrite(rotated_img_path, rotated_img)

    dilated = rotated_img.copy()
    start_char = []
    end_char = []

    # Logic for header line removal
    """So the approach is to find the row which is having the maximum number of 
        white pixels and then convert it and all rows near to it into black."""

    # A row vector of zeros with the same width as dilated
    row = np.zeros(dilated.shape[1])
    mxrow = 0  # var to track row with max white pixels
    mxcnt = 0  # var to track max white pixels
    kernel = np.ones((2, 2), np.uint8)

    # Apply morphological operation dilation and erosion which applying together helps to
    # remove noises
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)

    # For each row find no.of white pixels and update maxcnt and mxrow
    for i in range(dilated.shape[0]):
        cnt = 0
        for j in range(dilated.shape[1]):
            if dilated[i][j] == 255:
                cnt += 1
        if mxcnt < cnt:
            mxcnt = cnt
            mxrow = i

    print(dilated.shape[0])
    # plus: A variable to define a range above the mxrow, calculated as one-tenth of the image height.
    # from 0th row to (mxrow+plus)th row to black i.e zero which removes the header line
    plus = dilated.shape[0] // 10
    for i in range(0, mxrow + plus):
        dilated[i] = row

    cv2.imshow("HeaderLine Removed", dilated)
    cv2.waitKey(1000)
    header_img_path = os.path.join("./segmented_characters/", f"headerline_img.png")
    cv2.imwrite(header_img_path, dilated)

    ## Character Segmentation
    col_sum = np.zeros((dilated.shape[1]))
    col_sum = np.sum(dilated, axis=0)
    thresh = 0.08 * dilated.shape[0]

    for i in range(1, dilated.shape[1]):
        if col_sum[i - 1] <= thresh and col_sum[i] > thresh and col_sum[i + 1] > thresh:
            start_char.append(i)
        elif (
            col_sum[i - 1] > thresh
            and col_sum[i] <= thresh
            and col_sum[i + 1] <= thresh
        ):
            end_char.append(i)

    start_char.append(end_char[-1])
    character = []

    for i in range(1, len(start_char)):
        roi = rotated_img[:, start_char[i - 1] : start_char[i]]
        h = roi.shape[1]
        w = roi.shape[0]
        roi = helpers.extractroi(roi)
        roi = cv2.resize(roi, (180, 180))
        if helpers.check(roi) and h >= 30 and w >= 30:
            character.append(roi)

            cv2.imshow("CHARACTER_SEGMENTED", roi)
            cv2.waitKey(1000)

            char_img_path = os.path.join(output_dir, f"char_{i}.png")
            cv2.imwrite(char_img_path, roi)

    ls = []
    for char in character:
        pred = helpers.predictchar(char, model)
        ls.append(mapping[pred])

    return ls


def test():
    image_paths = ["./SampleImages/kamal2.jpeg"]
    output_dir = "./segmented_characters/"
    os.makedirs(output_dir, exist_ok=True)

    correct_answers = [""]
    score = 0
    multiplication_factor = 2

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        answer = predict(image_path, output_dir)
        print("".join(answer))
        if correct_answers[i] == answer:
            score += len(answer) * multiplication_factor

        print("Segmented characters saved in", output_dir)
        print("The final score of the participant is", score)


if __name__ == "__main__":
    test()
