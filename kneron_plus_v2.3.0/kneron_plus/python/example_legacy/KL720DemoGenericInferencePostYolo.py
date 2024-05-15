# ******************************************************************************
#  Copyright (c) 2021-2022. Kneron Inc. All rights reserved.                   *
# ******************************************************************************

import os
import sys
import argparse

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))
sys.path.insert(1, os.path.join(PWD, '../example'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
from utils.ExamplePostProcess import post_process_yolo_v5
import kp
import cv2

MODEL_FILE_PATH = os.path.join(PWD, '../../res/models/KL720/YoloV5s_640_640_3/models_720.nef')
IMAGE_FILE_PATH = os.path.join(PWD, '../../res/images/car_park_barrier_608x608.bmp')
LOOP_TIME = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL720 Demo Generic Inference with YOLO Post-Process Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    args = parser.parse_args()

    usb_port_id = args.port_id

    """
    check device USB speed (Recommend run KL720 at super speed)
    """
    try:
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Warning] Device is not run at super speed.' + '\033[0m')
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)

    """
    connect the device
    """
    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                     str(exception)))
        exit(0)

    """
    setting timeout of the usb communication with the device
    """
    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=5000)
    print(' - Success')

    """
    upload model to device
    """
    try:
        print('[Upload Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=MODEL_FILE_PATH)
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    prepare the image
    """
    print('[Read Image]')
    img = cv2.imread(filename=IMAGE_FILE_PATH)
    img_bgr565 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2BGR565)
    print(' - Success')

    """
    prepare generic raw inference image descriptor
    """
    generic_raw_image_header = kp.v1.GenericRawImageHeader(
        model_id=model_nef_descriptor.models[0].id,
        resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
        padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
        normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
        inference_number=0
    )

    """
    starting inference work
    """
    print('[Starting Inference Work]')
    print(' - Starting inference loop {} times'.format(LOOP_TIME))
    print(' - ', end='')
    for i in range(LOOP_TIME):
        try:
            kp.v1.inference.generic_raw_inference_send(device_group=device_group,
                                                       generic_raw_image_header=generic_raw_image_header,
                                                       image=img_bgr565,
                                                       image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)

            generic_raw_result = kp.v1.inference.generic_raw_inference_receive(device_group=device_group,
                                                                               generic_raw_image_header=generic_raw_image_header,
                                                                               model_nef_descriptor=model_nef_descriptor)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        print('.', end='', flush=True)
    print()

    """
    retrieve inference node output 
    """
    print('[Retrieve Inference Node Output ]')
    inf_node_output_list = []
    for node_idx in range(generic_raw_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                         generic_raw_result=generic_raw_result,
                                                                                         channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        inf_node_output_list.append(inference_float_node_output)

    print(' - Success')

    """
    post-process the last raw output
    """
    print('[Yolo V5s Post-Processing]')
    yolo_result = post_process_yolo_v5(
        inference_float_node_output_list=inf_node_output_list,
        hardware_preproc_info=generic_raw_result.header.hw_pre_proc_info,
        thresh_value=0.15,
        with_sigmoid=False)

    print(' - Success')

    print('[Result]')
    print(yolo_result)

    """
    output result image
    """
    print('[Output Result Image]')
    output_img_name = 'output_{}'.format(os.path.basename(IMAGE_FILE_PATH))
    print(' - Output bounding boxes on \'{}\''.format(output_img_name))
    for yolo_box_result in yolo_result.box_list:
        b = 100 + (25 * yolo_box_result.class_num) % 156
        g = 100 + (80 + 40 * yolo_box_result.class_num) % 156
        r = 100 + (120 + 60 * yolo_box_result.class_num) % 156
        color = (b, g, r)

        cv2.rectangle(img=img,
                      pt1=(int(yolo_box_result.x1), int(yolo_box_result.y1)),
                      pt2=(int(yolo_box_result.x2), int(yolo_box_result.y2)),
                      color=color)
    cv2.imwrite(os.path.join(PWD, './{}'.format(output_img_name)), img=img)
