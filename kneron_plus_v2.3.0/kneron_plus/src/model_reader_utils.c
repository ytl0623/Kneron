#include <stdio.h>
#include "internal_func.h"
#include "kp_struct.h"

#define err_print(format, ...) { printf(format, ##__VA_ARGS__); fflush(stdout); }

/******************************************************************
 * utils
 ******************************************************************/

int is_tensor_info_reallocted(kp_tensor_descriptor_t* tensor) {
    if ((NULL == tensor->name) ||
        (0 < tensor->shape_npu_len &&
        NULL == tensor->shape_npu) ||
        (0 < tensor->shape_onnx_len &&
        NULL == tensor->shape_onnx) ||
        (0 < tensor->quantization_parameters.quantized_fixed_point_descriptor_num &&
        NULL == tensor->quantization_parameters.quantized_fixed_point_descriptor)) {
        err_print("construct nef info in node tensor info fail: realloc memory fail ...\n");
        return KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
    }

    return KP_SUCCESS;
}

uint32_t convert_data_format_to_kp_tensor_format(uint32_t data_format, uint32_t target_chip) {
    if (KP_MODEL_TARGET_CHIP_KL520 == target_chip) {
        switch (data_format)
        {
        case DATA_FMT_KL520_4W4C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B;
            break;
        case DATA_FMT_KL520_16W1C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B;
            break;
        }
    } else if (KP_MODEL_TARGET_CHIP_KL720 == target_chip) {
        switch (data_format)
        {
        case DATA_FMT_KL720_4W4C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B;
            break;
        case DATA_FMT_KL720_16W1C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B;
            break;
        case DATA_FMT_KL720_1W16C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B;
            break;
        case DATA_FMT_KL720_8W1C16B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B;
            break;
        }
    } else if (KP_MODEL_TARGET_CHIP_KL530 == target_chip) {
        switch (data_format)
        {
        case DATA_FMT_KL530_4W4C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B;
            break;
        case DATA_FMT_KL530_16W1C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B;
            break;
        case DATA_FMT_KL530_1W16C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B;
            break;
        case DATA_FMT_KL530_8W1C16B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B;
            break;
        }
    } else if (KP_MODEL_TARGET_CHIP_KL630 == target_chip) {
        switch (data_format)
        {
        case DATA_FMT_KL630_4W4C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B;
            break;
        case DATA_FMT_KL630_16W1C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B;
            break;
        case DATA_FMT_KL630_1W16C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B;
            break;
        case DATA_FMT_KL630_8W1C16B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B;
            break;
        }
    } else if (KP_MODEL_TARGET_CHIP_KL730 == target_chip) {
        switch (data_format)
        {
        case DATA_FMT_KL730_4W4C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B;
            break;
        case DATA_FMT_KL730_16W1C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B;
            break;
        case DATA_FMT_KL730_1W16C8B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B;
            break;
        case DATA_FMT_KL730_8W1C16B:
            return KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B;
            break;
        }
    }

    return KP_MODEL_TENSOR_DATA_LAYOUT_UNKNOWN;
}
