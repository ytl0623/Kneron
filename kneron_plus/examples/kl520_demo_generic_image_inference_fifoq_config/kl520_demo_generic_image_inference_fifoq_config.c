/**
 * @file        kl520_demo_generic_image_inference_fifoq_config.c
 * @brief       main code of generic image inference with customized fifoq configuration
 * @version     0.1
 * @date        2022-06-15
 *
 * @copyright   Copyright (c) 2022 Kneron Inc. All rights reserved.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kp_core.h"
#include "kp_inference.h"
#include "helper_functions.h"

static char _scpu_fw_path[128] = "../../res/firmware/KL520/fw_scpu.bin";
static char _ncpu_fw_path[128] = "../../res/firmware/KL520/fw_ncpu.bin";
static char _model_file_path[128] = "../../res/models/KL520/tiny_yolo_v3/models_520.nef";
static char _image_file_path[128] = "../../res/images/bike_cars_street_224x224.bmp";
static int _loop = 10;

static kp_devices_list_t *_device_list;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_generic_image_inference_desc_t _input_data;
static kp_generic_image_inference_result_header_t _output_desc;
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
    ret = kp_load_firmware_from_file(_device, _scpu_fw_path, _ncpu_fw_path);
    printf("upload firmware ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

     /******* prepare attribute for setting fifo queue size *******/
    kp_ddr_manage_attr_t ddr_attr;

    ddr_attr.model_size = 0; // The attributes with 0 value will be auto adjust to the suitable count or size

    // (640 * 480 * 2) is the size for 640 x 480 RGB565 image and image below this size
    // 1024 is reserved for header size
    ddr_attr.input_buffer_size = (640 * 480 * 2) + 1024;
    ddr_attr.input_buffer_count = 2;

    // (2 * 1024 * 1024) is for at most 2 MB raw output data
    // 1024 is reserved for header size
    ddr_attr.result_buffer_size = (2 * 1024 * 1024) + 1024;
    ddr_attr.result_buffer_count = 0; // The attributes with 0 value will be auto adjust to the suitable count or size

    // The attribute must be stored to kp_device_group_t before load model
    ret = kp_store_ddr_manage_attr(_device, ddr_attr);
    printf("store attribute ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* upload model to device *******/
    // ddr attribute will be set to device while load model to device
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    printf("upload model ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* allocate memory for raw output *******/
    // by default here select first model
    uint32_t raw_buf_size = _model_desc.models[0].max_raw_out_size;
    uint8_t *raw_output_buf = (uint8_t *)malloc(raw_buf_size);

    _input_data.model_id = _model_desc.models[0].id;    // first model ID
    _input_data.inference_number = 0;                   // inference number, used to verify with output result
    _input_data.num_input_node_image = 1;               // number of image

    /******* prepare the image buffer read from file *******/
    // here convert a bmp file to RGB565 format buffer
    _img_buf = helper_bmp_file_to_raw_buffer(_image_file_path, &_img_width, &_img_height, KP_IMAGE_FORMAT_RGB565);
    printf("read image ... %s\n", (_img_buf) ? "OK" : "failed");

    /******* set up the input descriptor *******/
    _input_data.input_node_image_list[0].resize_mode = KP_RESIZE_ENABLE;        // enable resize in pre-process
    _input_data.input_node_image_list[0].padding_mode = KP_PADDING_CORNER;      // enable corner padding in pre-process
    _input_data.input_node_image_list[0].normalize_mode = KP_NORMALIZE_KNERON;  // this depends on models
    _input_data.input_node_image_list[0].image_format = KP_IMAGE_FORMAT_RGB565; // image format
    _input_data.input_node_image_list[0].width = _img_width;                    // image width
    _input_data.input_node_image_list[0].height = _img_height;                  // image height
    _input_data.input_node_image_list[0].crop_count = 0;                        // number of crop area, 0 means no cropping
    _input_data.input_node_image_list[0].image_buffer = (uint8_t *)_img_buf;    // buffer of image data

    printf("\nstarting inference loop %d times:\n", _loop);

    /******* starting inference work *******/
    for (int i = 0; i < _loop; i++)
    {
        ret = kp_generic_image_inference_send(_device, &_input_data);
        if (ret != KP_SUCCESS)
            break;

        ret = kp_generic_image_inference_receive(_device, &_output_desc, raw_output_buf, raw_buf_size);
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

    printf("\ninference loop is done\n");

    kp_inf_float_node_output_t *output_nodes[2] = {NULL};

    // retrieve output nodes in floating point format
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf, KP_CHANNEL_ORDERING_HCW);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf, KP_CHANNEL_ORDERING_HCW);

    // print and save the content of node data to files
    helper_dump_floating_node_data_to_files(output_nodes, _output_desc.num_output_node, _image_file_path);

    free(output_nodes[0]);
    free(output_nodes[1]);

    free(raw_output_buf);

    return 0;
}
