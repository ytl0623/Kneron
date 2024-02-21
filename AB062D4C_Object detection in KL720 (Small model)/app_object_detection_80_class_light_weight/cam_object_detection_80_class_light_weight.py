# ******************************************************************************
#  Copyright (c) 2021. Kneron Inc. All rights reserved.                        *
# ******************************************************************************

import sys
import os
import argparse
import kp_yolo as kp
import cv2
import json
import time

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '../..'))
os.chdir(PWD)

from utils.ui import load_logo, render_logo, render_menu, render_fps, render_box
from utils.helper import check_device
from class_table import CLASS_TABLE

DEVICE_TYPE = ''
SCPU_FW_PATH = ''
NCPU_FW_PATH = ''
MODEL_FILE_PATH = ''
LOGO_IMG = None
LOGO_IMG_MASK = None

INF_RESOLUTION_WIDTH = 640
INF_RESOLUTION_HEIGHT = 640
MODEL_NEF_DESCRIPTOR = None


def draw_result(det_res, image):
    # for multiple pedestrian
    for yolo_result in det_res.box_list:
        b = 100 + (25 * yolo_result.class_num) % 156
        g = 100 + (80 + 40 * yolo_result.class_num) % 156
        r = 100 + (120 + 60 * yolo_result.class_num) % 156
        color = (b, g, r)

        render_box(image=image,
                   x1=yolo_result.x1,
                   y1=yolo_result.y1,
                   x2=yolo_result.x2,
                   y2=yolo_result.y2,
                   text='{} {:.4f}'.format(CLASS_TABLE[yolo_result.class_num].title(), yolo_result.score),
                   color=color)


def init_proj_config(config_path):
    global DEVICE_TYPE
    global SCPU_FW_PATH
    global NCPU_FW_PATH
    global MODEL_FILE_PATH
    global LOGO_IMG, LOGO_IMG_MASK

    with open(config_path, newline='') as file:
        config_dict = json.load(file)
        DEVICE_TYPE = config_dict['device_type']
        SCPU_FW_PATH = os.path.abspath(os.path.join(PWD, config_dict['scpu_fw_path']))
        NCPU_FW_PATH = os.path.abspath(os.path.join(PWD, config_dict['ncpu_fw_path']))
        MODEL_FILE_PATH = os.path.abspath(os.path.join(PWD, config_dict['nef_model_path']))
        LOGO_IMG, LOGO_IMG_MASK = load_logo(logo_png_path=os.path.abspath(os.path.join(PWD, config_dict['logo_img_path'])))


def init_device(usb_port_id):
    global MODEL_NEF_DESCRIPTOR

    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])

        print('[Set Device Timeout]')
        kp.core.set_timeout(device_group=device_group, milliseconds=5000)

        print('[Update Firmware] - Do not unplug AI dongle when update firmware...')
        kp.core._update_kdp2_firmware_from_file(device_group=device_group,
                                                scpu_fw_path=SCPU_FW_PATH,
                                                ncpu_fw_path=NCPU_FW_PATH,
                                                auto_reboot=True)

        print('[Upload Model]')
        MODEL_NEF_DESCRIPTOR = kp.core.load_model_from_file(device_group=device_group, file_path=MODEL_FILE_PATH)

    except kp.ApiKPException as exception:
        print('Error: \'{}\''.format(str(exception)))
        exit(0)

    return device_group


def main(usb_port_id):
    global LOGO_IMG, LOGO_IMG_MASK, MODEL_NEF_DESCRIPTOR
    _try_cap_time = 0
    _loop = 0
    _fps = 0

    # preparing device
    device_group = init_device(usb_port_id)

    # prepare app yolo inference config
    print('[Configure YOLO Application]')
    app_yolo_config = kp.AppYoloConfig(
        model_id=MODEL_NEF_DESCRIPTOR.models[0].id,
        model_norm=kp.NormalizeMode.KP_NORMALIZE_KNERON
    )

    print('[Preparing Camera] ...')
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open camera device.")
        exit(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, INF_RESOLUTION_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, INF_RESOLUTION_HEIGHT)

    origin_image_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    origin_image_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # prepare resize & padding parameters
    resize_ratio = 1.0
    if (origin_image_width != INF_RESOLUTION_WIDTH) or (origin_image_height != INF_RESOLUTION_HEIGHT):
        if (origin_image_width / INF_RESOLUTION_WIDTH) > (origin_image_height / INF_RESOLUTION_HEIGHT):
            resize_ratio = INF_RESOLUTION_WIDTH / origin_image_width
        else:
            resize_ratio = INF_RESOLUTION_HEIGHT / origin_image_height

    image_width = int(origin_image_width * resize_ratio)
    image_height = int(origin_image_height * resize_ratio)

    image_width_pad = int((INF_RESOLUTION_WIDTH - image_width) / 2)
    image_height_pad = int((INF_RESOLUTION_HEIGHT - image_height) / 2)

    print('[Starting Inference Work] ...')
    while True:
        time_start = time.time()

        ret, img = capture.read()

        if not ret and _try_cap_time < 10:
            _try_cap_time += 1
            time.sleep(0.1)
            continue
        elif not ret:
            print('Error: opencv open camera failed')
            break

        # do resize & padding to fit model aspect ratio
        img = cv2.resize(img, (image_width, image_height))
        img = cv2.copyMakeBorder(img,
                                 image_height_pad,
                                 image_height_pad,
                                 image_width_pad,
                                 image_width_pad,
                                 cv2.BORDER_CONSTANT,
                                 (0, 0, 0))

        # convert to rgb565
        img_bgr565 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2BGR565)

        # inference
        try:
            kp.inference.app_yolo_inference_send(device_group=device_group,
                                                 inference_number=-1,
                                                 app_yolo_config=app_yolo_config,
                                                 image=img_bgr565,
                                                 image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)
            _, res_data = kp.inference.app_yolo_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print('Error: inference failed, error = {}'.format(exception))
            break

        # draw inference result and UI info
        if res_data.box_count != 0:
            draw_result(res_data, img)

        time_end = time.time()

        _loop += 1

        if 30 == _loop:
            _fps = 1 / (time_end - time_start)
            _loop = 0

        render_fps(image=img, fps=_fps)
        render_logo(image=img, logo_image=LOGO_IMG, logo_image_mask=LOGO_IMG_MASK)
        render_menu(image=img,
                    menu_list=['\'q\' or \'Q\' : Exit'])

        # show 
        cv2.imshow('80 Class Object Detection (Light Weight Model)', img)

        key = cv2.waitKey(1)
        if (key == ord('q')) or key == ord('Q'):
            break

    capture.release()
    cv2.destroyWindow("80 Class Object Detection (Light Weight Model)")

    print('[Stop]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='80 Class Object Detection (Light Weight Model) App Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    args = parser.parse_args()

    init_proj_config("config.json")
    check_device(usb_port_id=args.port_id, device_type=DEVICE_TYPE)
    main(args.port_id)
