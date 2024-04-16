/**
 * @file        kl720_demo_generic_inference_bypass_pre_proc.c
 * @brief       main code of generic inference (output raw data) with bypassing pre-process function
 * @version     0.1
 * @date        2021-09-08
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kp_core.h"
#include "legacy/kp_inference_v1.h"
#include "helper_functions.h"

static char _model_file_path[128] = "../../res/models/KL720/YoloV5s_640_640_3/models_720.nef";
static char _image_file_path[128] = "../../res/images/people_talk_in_street_640x640_rgba8888_normalized.bin";
static int _loop = 100;

static kp_devices_list_t *_device_list;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_generic_raw_bypass_pre_proc_image_header_t _input_desc;
static kp_generic_raw_bypass_pre_proc_result_header_t _output_desc;
static char *_img_buf;
static int _img_width = 640;
static int _img_height = 640;

int main(int argc, char *argv[])
{
    // each device has a unique port ID, 0 for auto-search
    int port_id = (argc > 1) ? atoi(argv[1]) : 0;
    int ret;

    /******* check the device USB speed *******/
    int link_speed;
    _device_list = kp_scan_devices();

    helper_get_device_usb_speed_by_port_id(_device_list, port_id, &link_speed);
    if (KP_USB_SPEED_SUPER != link_speed)
        printf("[warning] device is not run at super speed.\n");

    /******* connect the device *******/
    _device = kp_connect_devices(1, &port_id, NULL);
    printf("connect device ... %s\n", (_device) ? "OK" : "failed");

    if (NULL == _device) {
        return -1;
    }

    kp_set_timeout(_device, 5000); // 5 secs timeout

    /******* upload model to device *******/
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    printf("upload model ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* allocate memory for raw output *******/
    // by default here select first model
    uint32_t raw_buf_size = _model_desc.models[0].max_raw_out_size;
    uint8_t *raw_output_buf = (uint8_t *)malloc(raw_buf_size);

    /******* prepare the image buffer read from file *******/
    // here convert a bmp file to RGB565 format buffer
    _img_buf = helper_bin_file_to_raw_buffer(_image_file_path, _img_width, _img_height, KP_IMAGE_FORMAT_RGBA8888);
    printf("read image ... %s\n", (_img_buf) ? "OK" : "failed");

    /******* set up the input descriptor *******/
    _input_desc.model_id = _model_desc.models[0].id;    // first model ID
    _input_desc.inference_number = 0;                   // inference number, used to verify with output result
    _input_desc.image_buffer_size = _img_width * _img_height * 4;

    printf("\nstarting inference loop %d times:\n", _loop);

    /******* starting inference work *******/
    for (int i = 0; i < _loop; i++)
    {
        ret = kp_generic_raw_inference_bypass_pre_proc_send(_device, &_input_desc, (uint8_t *)_img_buf);
        if (ret != KP_SUCCESS)
            break;

        ret = kp_generic_raw_inference_bypass_pre_proc_receive(_device, &_output_desc, raw_output_buf, raw_buf_size);
        if (ret != KP_SUCCESS)
            break;

        printf(".");
        fflush(stdout);
    }
    printf("\n");

    free(_img_buf);
    kp_release_model_nef_descriptor(&_model_desc);
    kp_disconnect_devices(_device);

    if (ret != KP_SUCCESS)
    {
        printf("\ninference failed, error = %d (%s)\n", ret, kp_error_string(ret));
        return -1;
    }

    printf("\ninference loop is done, starting post-processing ...\n");

    kp_inf_float_node_output_t *output_nodes[3] = {NULL}; // tiny yolo v5 outputs 3 nodes, described by _output_desc.num_output_node

    // retrieve output nodes in floating point format
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf, KP_CHANNEL_ORDERING_CHW);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf, KP_CHANNEL_ORDERING_CHW);
    output_nodes[2] = kp_generic_inference_retrieve_float_node(2, raw_output_buf, KP_CHANNEL_ORDERING_CHW);

    // print and save the content of node data to files
    helper_dump_floating_node_data_to_files(output_nodes, _output_desc.num_output_node, _image_file_path);

    free(output_nodes[0]);
    free(output_nodes[1]);
    free(output_nodes[2]);

    free(raw_output_buf);

    return 0;
}
