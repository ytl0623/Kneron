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

def get_palette(mapping, new=False, seed=9487):
    np.random.seed(seed)
    return [list(np.random.choice(range(256), size=3))
            for _ in range(mapping)]

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
    parser = argparse.ArgumentParser(description='KL720 Kneron Model Zoo Generic Inference Example - STDC.')
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


o_im = cv2.imread(filename=image_file_path)

# change output array data order from nchw to hwc
pred = inf_node_output_list[0].ndarray.squeeze().transpose(1,2,0) # should only one output node

# channel number means all possible class number
n_c = pred.shape[2] 

# upscaling inference result array to origin image size
pred = cv2.resize(pred, (o_im.shape[1], o_im.shape[0]), interpolation=cv2.INTER_LINEAR)

# find max score class
pred = pred.argmax(2) 


"""
output result image
"""
colors = get_palette(n_c)
seg_res_vis = np.zeros(o_im.shape, np.uint8)
for c in range(n_c):
    seg_res_vis[pred == c] = colors[c]

print('[Output Result Image]')
output_img_name = 'output_{}'.format(os.path.basename(image_file_path))

print(' - Output Segmentation result on \'{}\''.format(output_img_name))
cv2.imwrite(output_img_name, seg_res_vis)

