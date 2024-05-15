/**
 * @file        kl520_demo_generic_inference_crop.c
 * @brief       main code of generic inference (output raw data) with crop boxes specified
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
#include "kp_inference.h"
#include "helper_functions.h"
#include "postprocess.h"

static char _scpu_fw_path[128] = "../../res/firmware/KL520/fw_scpu.bin";
static char _ncpu_fw_path[128] = "../../res/firmware/KL520/fw_ncpu.bin";
static char _model_file_path[128] = "../../res/models/KL520/tiny_yolo_v3/models_520.nef";
static char _image_file_path[128] = "../../res/images/one_bike_many_cars_800x800.bmp";
static int _loop = 50;

static kp_devices_list_t *_device_list;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_generic_image_inference_desc_t _input_data;
static kp_generic_image_inference_result_header_t _output_desc_0;
static kp_generic_image_inference_result_header_t _output_desc_1;
static char *_img_buf;
static int _img_width, _img_height;

int main(int argc, char *argv[])
{
    // each device has a unique port ID, 0 for auto-search
    int port_id = (argc > 1) ? atoi(argv[1]) : 0;

    /******* check the device USB speed *******/
    int link_speed;
    _device_list = kp_scan_devices();
    helper_get_device_usb_speed_by_port_id(_device_list, port_id, &link_speed);
    if (KP_USB_SPEED_HIGH != link_speed)
        printf("[warning] device is not run at high speed.\n");

    /******* connect the device *******/
    _device = kp_connect_devices(1, &port_id, NULL);
    printf("connect device ... %s\n", (_device) ? "OK" : "failed");

    if (NULL == _device) {
        return -1;
    }

    kp_set_timeout(_device, 5000); // 5 secs timeout

    /******* upload firmware to device *******/
    int ret = kp_load_firmware_from_file(_device, _scpu_fw_path, _ncpu_fw_path);
    printf("upload firmware ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* upload model to device *******/
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    printf("upload model ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* allocate memory for raw output *******/
    // by default here select first model
    uint32_t raw_buf_size = _model_desc.models[0].max_raw_out_size;
    uint8_t *raw_output_buf_crop_box_0 = (uint8_t *)malloc(raw_buf_size);
    uint8_t *raw_output_buf_crop_box_1 = (uint8_t *)malloc(raw_buf_size);

    /******* prepare the image buffer read from file *******/
    // here convert a bmp file to RGB565 format buffer
    _img_buf = helper_bmp_file_to_raw_buffer(_image_file_path, &_img_width, &_img_height, KP_IMAGE_FORMAT_RGB565);
    printf("read image ... %s\n", (_img_buf) ? "OK" : "failed");

    /******* set up the input descriptor *******/
    _input_data.model_id = _model_desc.models[0].id;    // first model ID
    _input_data.inference_number = 0;                   // inference number, used to verify with output result
    _input_data.num_input_node_image = 1;               // number of image

    _input_data.input_node_image_list[0].resize_mode = KP_RESIZE_ENABLE;        // enable resize in pre-process
    _input_data.input_node_image_list[0].padding_mode = KP_PADDING_CORNER;      // enable corner padding in pre-process
    _input_data.input_node_image_list[0].normalize_mode = KP_NORMALIZE_KNERON;  // this depends on models
    _input_data.input_node_image_list[0].image_format = KP_IMAGE_FORMAT_RGB565; // image format
    _input_data.input_node_image_list[0].width = _img_width;                    // image width
    _input_data.input_node_image_list[0].height = _img_height;                  // image height
    _input_data.input_node_image_list[0].crop_count = 2;                        // number of crop area, 0 means no cropping
    _input_data.input_node_image_list[0].inf_crop[0].crop_number = 0;           // crop number for sync, could be specified by user
    _input_data.input_node_image_list[0].inf_crop[0].x1 = 0;
    _input_data.input_node_image_list[0].inf_crop[0].y1 = 0;
    _input_data.input_node_image_list[0].inf_crop[0].width = 400;
    _input_data.input_node_image_list[0].inf_crop[0].height = 400;
    _input_data.input_node_image_list[0].inf_crop[1].crop_number = 1; // crop number for sync, could be specified by user
    _input_data.input_node_image_list[0].inf_crop[1].x1 = 230;
    _input_data.input_node_image_list[0].inf_crop[1].y1 = 335;
    _input_data.input_node_image_list[0].inf_crop[1].width = 450;
    _input_data.input_node_image_list[0].inf_crop[1].height = 450;
    _input_data.input_node_image_list[0].image_buffer = (uint8_t *)_img_buf;    // buffer of image data

    printf("\nstarting inference loop %d times:\n", _loop);

    /******* starting inference work *******/
    for (int i = 0; i < _loop; i++)
    {
        ret = kp_generic_image_inference_send(_device, &_input_data);
        if (ret != KP_SUCCESS)
            break;

        ret = kp_generic_image_inference_receive(_device, &_output_desc_0, raw_output_buf_crop_box_0, raw_buf_size);
        if (ret != KP_SUCCESS)
            break;

        // make sure crop box number is correct
        if (_output_desc_0.crop_number != 0)
            printf("\nincorrect crop_number %d, expected crop_number %d at frame %d\n", _output_desc_0.crop_number, 0, i);

        ret = kp_generic_image_inference_receive(_device, &_output_desc_1, raw_output_buf_crop_box_1, raw_buf_size);
        if (ret != KP_SUCCESS)
            break;

        // make sure crop box number is correct
        if (_output_desc_1.crop_number != 1)
            printf("\nincorrect crop_number %d, expected crop_number %d at frame %d\n", _output_desc_1.crop_number, 1, i);

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

    printf("\ndoing tiny yolo v3 post-processing ...\n");

    kp_yolo_result_t *yolo_result = (kp_yolo_result_t *)malloc(sizeof(kp_yolo_result_t));

    kp_inf_float_node_output_t *output_nodes[2] = {NULL}; // tiny yolo v3 outputs only two nodes, described by _output_desc.num_output_node

    // retrieve output nodes in floating point format in first crop box
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf_crop_box_0, KP_CHANNEL_ORDERING_HCW);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf_crop_box_0, KP_CHANNEL_ORDERING_HCW);

    // post-process yolo v3 output nodes to class/bounding boxes
    post_process_yolo_v3(output_nodes, _output_desc_0.num_output_node, &_output_desc_0.pre_proc_info[0], 0.2, yolo_result);

    helper_print_yolo_box_of_crop_area_on_bmp(yolo_result, _image_file_path, _input_data.input_node_image_list[0].inf_crop[0]);

    // retrieve output nodes in floating point format in second crop box
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf_crop_box_1, KP_CHANNEL_ORDERING_HCW);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf_crop_box_1, KP_CHANNEL_ORDERING_HCW);

    // post-process yolo v3 output nodes to class/bounding boxes
    post_process_yolo_v3(output_nodes, _output_desc_1.num_output_node, &_output_desc_1.pre_proc_info[0], 0.2, yolo_result);

    helper_print_yolo_box_of_crop_area_on_bmp(yolo_result, _image_file_path, _input_data.input_node_image_list[0].inf_crop[1]);

    free(output_nodes[0]);
    free(output_nodes[1]);
    free(yolo_result);

    free(raw_output_buf_crop_box_0);
    free(raw_output_buf_crop_box_1);

    return 0;
}
