# ******************************************************************************
#  Copyright (c) 2022. Kneron Inc. All rights reserved.                        *
# ******************************************************************************

import os
import sys
import argparse

import kp
import cv2
import numpy as np
import math
from utils import NEF_Parser_lite

LOOP_TIME = 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def convert_numpy_to_rgba_and_width_align_4(data):
    """Converts the numpy data into RGBA.

    720 input is 4 byte width aligned.

    """

    height, width, channel = data.shape

    width_aligned = 4 * math.ceil(width / 4.0)
    aligned_data = np.zeros((height, width_aligned, 4), dtype=np.int8)
    aligned_data[:height, :width, :channel] = data
    aligned_data = aligned_data.flatten()
    
    return aligned_data.tobytes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL720 Kneron Model Zoo Generic Inference Example - RegNetX.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    parser.add_argument('-img',
                        '--img_path',
                        help='input image path',
                        type=str)
    parser.add_argument('-nef',
                        '--nef_model_path',
                        help='input NEF model path',
                        type=str)

    args = parser.parse_args()

    assert args.img_path is not None, "need to set input image but got None"
    assert args.nef_model_path is not None, "need to set nef model path but got None"

    usb_port_id = args.port_id
    nef_model_path = args.nef_model_path
    image_file_path = args.img_path


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
    check PLUS version
    """
    assert kp.core.get_version() == '1.3.0', f"Only verified on PLUS version 1.3.0, but got {kp.core.get_version()}"

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
                                                            file_path=nef_model_path)
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    extract input radix from NEF
    """
    nef_radix = NEF_Parser_lite.extract_input_radix_from_kl720_nef(nef_model_path) # only support single model NEF

    """
    prepare the image
    """
    nef_model_width = model_nef_descriptor.models[0].width
    nef_model_height = model_nef_descriptor.models[0].height
    print('[Read Image]')
    img = cv2.imread(filename=image_file_path)
    img_height, img_width, img_channels = img.shape

    # resize to model input size
    img = cv2.resize(img, ( nef_model_width,  nef_model_height), interpolation=cv2.INTER_AREA)

    # to rgb
    img_input = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    # this model trained with normalize method: (data - 128)/256 , 
    img_input = img_input/256.
    img_input -= 0.5

    # toolchain calculate the radix value from input data (after normalization), and set it into NEF model.
    # NPU will divide input data "2^radix" automaticly, so, we have to scaling the input data here due to this reason.
    img_input *= (1 << nef_radix)

    # convert rgb to rgba and width align 4, due to npu requirement.
    img_buffer = convert_numpy_to_rgba_and_width_align_4(img_input)

    print(' - Success')

    """
    prepare app generic inference config
    """
    generic_raw_image_header = kp.GenericRawBypassPreProcImageHeader(
        model_id=model_nef_descriptor.models[0].id,
        image_buffer_size=len(img_buffer),
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
            kp.inference.generic_raw_inference_bypass_pre_proc_send(
                device_group=device_group,
                generic_raw_image_header=generic_raw_image_header,
                image_buffer=img_buffer)

            generic_raw_bypass_pre_proc_result = kp.inference.generic_raw_inference_bypass_pre_proc_receive(
                device_group=device_group,
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
    for node_idx in range(generic_raw_bypass_pre_proc_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=generic_raw_bypass_pre_proc_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
        inf_node_output_list.append(inference_float_node_output)

    print(' - Success')

# get class id with highest score
pred = inf_node_output_list[0].ndarray.squeeze() # should only one output node
pred = softmax(pred)
cls_num = np.argmax(pred)

print("[Result]")
print(" - Top-1 class id : " + str(cls_num) + "  (imagenet 1000 classes idx)")

