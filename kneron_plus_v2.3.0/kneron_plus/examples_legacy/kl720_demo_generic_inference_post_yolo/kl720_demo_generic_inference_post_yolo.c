/**
 * @file        kl720_demo_generic_inference_post_yolo.c
 * @brief       main code of generic inference (output raw data) with demo of post process function
 * @version     0.1
 * @date        2021-03-22
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
#include "postprocess.h"

static char _model_file_path[128] = "../../res/models/KL720/YoloV5s_640_640_3/models_720.nef";
static char _image_file_path[128] = "../../res/images/car_park_barrier_608x608.bmp";
static int _loop = 100;

static kp_devices_list_t *_device_list;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_generic_raw_image_header_t _input_desc;
static kp_generic_raw_result_header_t _output_desc;
static char *_img_buf;
static int _img_width, _img_height;

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
    _img_buf = helper_bmp_file_to_raw_buffer(_image_file_path, &_img_width, &_img_height, KP_IMAGE_FORMAT_RGB565);
    printf("read image ... %s\n", (_img_buf) ? "OK" : "failed");

    /******* set up the input descriptor *******/
    _input_desc.model_id = _model_desc.models[0].id;    // first model ID
    _input_desc.resize_mode = KP_RESIZE_ENABLE;        // enable resize in pre-process
    _input_desc.padding_mode = KP_PADDING_CORNER;      // enable corner padding in pre-process
    _input_desc.normalize_mode = KP_NORMALIZE_KNERON;  // this depends on models
    _input_desc.image_format = KP_IMAGE_FORMAT_RGB565; // image format
    _input_desc.width = _img_width;                    // image width
    _input_desc.height = _img_height;                  // image height
    _input_desc.inference_number = 0;                  // inference number, used to verify with output result
    _input_desc.crop_count = 0;                        // number of crop area, 0 means no cropping

    printf("\nstarting inference loop %d times:\n", _loop);

    /******* starting inference work *******/
    for (int i = 0; i < _loop; i++)
    {
        ret = kp_generic_raw_inference_send(_device, &_input_desc, (uint8_t *)_img_buf);
        if (ret != KP_SUCCESS)
            break;

        ret = kp_generic_raw_inference_receive(_device, &_output_desc, raw_output_buf, raw_buf_size);
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

    printf("\ndoing yolo v5 post-processing ...\n");

    kp_yolo_result_t *yolo_result = (kp_yolo_result_t *)malloc(sizeof(kp_yolo_result_t));

    kp_inf_float_node_output_t *output_nodes[3] = {NULL}; // tiny yolo v5 outputs 3 nodes, described by _output_desc.num_output_node

    // retrieve output nodes in floating point format in first crop box
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf, KP_CHANNEL_ORDERING_CHW);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf, KP_CHANNEL_ORDERING_CHW);
    output_nodes[2] = kp_generic_inference_retrieve_float_node(2, raw_output_buf, KP_CHANNEL_ORDERING_CHW);

    // post-process yolo v5 output nodes to class/bounding boxes
    post_process_yolo_v5_720(output_nodes, _output_desc.num_output_node, &_output_desc.pre_proc_info, 0.15, yolo_result);

    helper_print_yolo_box_on_bmp(yolo_result, _image_file_path);

    free(output_nodes[0]);
    free(output_nodes[1]);
    free(output_nodes[2]);
    free(yolo_result);

    free(raw_output_buf);

    return 0;
}
