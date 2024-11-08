import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

if __name__ == '__main__':

    '''
    load movie file
    '''
    # TODO - add movie file name
    # safe = 'temp'
    # fall = 'temp'
    #
    # result = []
    # src = cv2.VideoCapture(safe)
    # video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    # # ここちょっとマズいかも　留意しといて
    # need_length = 1 + (src.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * frame_step
    # # need_length = 1 + (n_frames - 1) * frame_step
    #
    # if need_length > video_length:
    #     start = 0
    # else:
    #     max_start = video_length - need_length
    #     start = random.randint(0, max_start + 1)
    #
    # src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret, frame = src.read()
    # resulet.append(format_frames(frame, output_size))
    #
    # for _ in range(src.get(cv2.CAP_PROP_FRAME_COUNT)):
    #     for _ in range(frame_step):
    #

    # class num:3 // normal, fall down, stumble
    safe_mov = 'safe.mp4'
    fall_mov = 'fall.mp4'
    stumble_mov = 'stumble.mp4'

    # load
    safe = cv2.VideoCapture(safe_mov)
    fall = cv2.VideoCapture(fall_mov)
    stumble = cv2.VideoCapture(stumble_mov)

    # frame count
    safe_len = safe.get(cv2.CAP_PROP_FRAME_COUNT)
    fall_len = fall.get(cv2.CAP_PROP_FRAME_COUNT)
    stumble_len = stumble.get(cv2.CAP_PROP_FRAME_COUNT)

    # TODO - maybe it needs to preprocess
    # frame = tf.image.convert_image_dtype(frame, tf.float32)
    # frame = tf.image.resize_with_pad(frame, *output_size)


    '''
    Create learning and test dataset
    '''
    class_number = 3
    data = []
    label = []

    # total frame num
    data_num = safe_len + fall_len + stumble_len
    print(data_num)

    '''
    model construction
    '''



    '''
    model evaluate
    '''



    '''
    learning model save
    '''
