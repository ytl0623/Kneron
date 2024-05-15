/**
 * @file        kneron_kne_reader.c
 * @brief       NEF model related functions - read NEF
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#include "internal_func.h"
#include "kneron_kne_c_reader.h"
#include "kp_struct.h"
#include <stdio.h>
#include <stdint.h>

#define err_print(format, ...) { printf(format, ##__VA_ARGS__); fflush(stdout); }

/******************************************************************
 * [private] KNE model constructor utils
 ******************************************************************/

int construct_kne_single_tensor_info_quantization_parameters_flatbuffer(KneronKNE_QuantizationParameters_table_t quantization_parameters_flatbuffer, kp_quantization_parameters_t *quantization_parameters) {
    int status = KP_SUCCESS;
    KneronKNE_DataType_enum_t data_type_enum;
    flatbuffers_int8_vec_t radix_vec;
    flatbuffers_uint8_vec_t scale_vec;

    if (NULL == quantization_parameters_flatbuffer ||
        NULL == quantization_parameters) {
        err_print("construct nef single tensor information quantization parameters in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    /* currently, we limited the input/output tensor scale data type in float32 */
    data_type_enum = KneronKNE_QuantizationParameters_scale_type(quantization_parameters_flatbuffer);
    if (KneronKNE_DataType_Float != data_type_enum) {
        err_print("construct nef single tensor information quantization parameters in model_descriptor fail: the scale data type is %s (%d), but expect in %s (%d) ...\n", KneronKNE_DataType_name(data_type_enum), data_type_enum, KneronKNE_DataType_name(KneronKNE_DataType_Float), KneronKNE_DataType_Float);
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    quantization_parameters->quantized_fixed_point_descriptor_num   = KneronKNE_QuantizationParameters_scale_count(quantization_parameters_flatbuffer);
    quantization_parameters->quantized_fixed_point_descriptor       = realloc_quantized_fixed_point_descriptor_list(quantization_parameters->quantized_fixed_point_descriptor, quantization_parameters->quantized_fixed_point_descriptor_num);

    if ((0 < quantization_parameters->quantized_fixed_point_descriptor_num) &&
        (NULL == quantization_parameters->quantized_fixed_point_descriptor)) {
        err_print("construct nef single tensor information quantization parameters in model_descriptor fail: alloc memory fail ...\n");
        status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
        goto FUNC_OUT;
    }

    radix_vec = KneronKNE_QuantizationParameters_radix(quantization_parameters_flatbuffer);
    scale_vec = KneronKNE_QuantizationParameters_scale(quantization_parameters_flatbuffer);

    if ((flatbuffers_int8_vec_len(radix_vec) != quantization_parameters->quantized_fixed_point_descriptor_num) ||
        (flatbuffers_uint8_vec_len(scale_vec) != quantization_parameters->quantized_fixed_point_descriptor_num * sizeof(float))) {
        err_print("construct nef single tensor information quantization parameters in model_descriptor fail: the number of radix(%u)/scale(%u) is not match with scale count (%u) ...\n", (uint32_t)flatbuffers_int8_vec_len(radix_vec), (uint32_t)flatbuffers_uint8_vec_len(scale_vec), quantization_parameters->quantized_fixed_point_descriptor_num);
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    for (int idx = 0; idx < quantization_parameters->quantized_fixed_point_descriptor_num; idx++) {
        quantization_parameters->quantized_fixed_point_descriptor[idx].radix = (int32_t)radix_vec[idx];
        quantization_parameters->quantized_fixed_point_descriptor[idx].scale = *(float *)(&scale_vec[idx * sizeof(float)]);
    }

FUNC_OUT:
    return status;
}

int construct_kne_single_model_tensors_info(KneronKNE_Tensor_table_t tensor, uint32_t target_chip, kp_tensor_descriptor_t *tensor_descriptor)
{
    int status                                            = KP_SUCCESS;
    kp_quantization_parameters_t *quantization_parameters = NULL;
    flatbuffers_int32_vec_t shape_npu;
    flatbuffers_int8_vec_t inv_shape_intrp_dim;
    KneronKNE_QuantizationParameters_table_t quantization_parameters_flatbuffer;

    if ((NULL == tensor) ||
        (NULL == tensor_descriptor)) {
        err_print("construct nef single model tensors information in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    tensor_descriptor->name                 = strcpy_dst_realloc(tensor_descriptor->name, (char*)KneronKNE_Tensor_name(tensor));
    tensor_descriptor->data_layout          = convert_data_format_to_kp_tensor_format(KneronKNE_Tensor_format(tensor), target_chip);

    /* shape_npu parse */
    shape_npu                               = KneronKNE_Tensor_shape(tensor);
    tensor_descriptor->shape_npu_len        = flatbuffers_int32_vec_len(shape_npu);
    tensor_descriptor->shape_npu            = realloc_tensor_shape(tensor_descriptor->shape_npu, tensor_descriptor->shape_npu_len);
    memcpy(tensor_descriptor->shape_npu, shape_npu, tensor_descriptor->shape_npu_len * flatbuffers_int32__size());

    /* shape_onnx parse */
    /* shape_onnx is interpreted by inv_shape_intrp_dim and shape_npu information */
    inv_shape_intrp_dim                     = KneronKNE_Tensor_inv_shape_intrp_dim(tensor);
    tensor_descriptor->shape_onnx_len       = flatbuffers_int8_vec_len(inv_shape_intrp_dim);
    tensor_descriptor->shape_onnx           = realloc_tensor_shape(tensor_descriptor->shape_onnx, tensor_descriptor->shape_onnx_len);

    for (int idx = 0; idx < tensor_descriptor->shape_onnx_len; idx++) {
        tensor_descriptor->shape_onnx[idx] = tensor_descriptor->shape_npu[inv_shape_intrp_dim[idx]];
    }

    /* quantization information parse */
    quantization_parameters                 = &(tensor_descriptor->quantization_parameters);
    quantization_parameters_flatbuffer      = KneronKNE_Tensor_quantization(tensor);
    status                                  = construct_kne_single_tensor_info_quantization_parameters_flatbuffer(quantization_parameters_flatbuffer, quantization_parameters);

    if (KP_SUCCESS != status)
        goto FUNC_OUT;

    status                                  = is_tensor_info_reallocted(tensor_descriptor);

    if (KP_SUCCESS != status)
        goto FUNC_OUT;

FUNC_OUT:
    return status;
}

int construct_kne_single_model_input_tensor_info(KneronKNE_ModelHeader_table_t model_header, kp_single_model_descriptor_t *single_model_descriptor)
{
    int status                                  = KP_SUCCESS;
    kp_tensor_descriptor_t *tensor_descriptor   = NULL;
    KneronKNE_Tensor_vec_t tensor_vec;
    KneronKNE_Tensor_table_t tensor;

    if ((NULL == model_header) ||
        (NULL == single_model_descriptor)) {
        err_print("construct nef single model information inputs tensor in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    tensor_vec = KneronKNE_ModelHeader_inputs(model_header);
    if (NULL == tensor_vec) {
        err_print("construct nef single model information inputs tensor in model_descriptor fail: invalid flatbuffer ...\n");
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    single_model_descriptor->input_nodes_num    = KneronKNE_Tensor_vec_len(tensor_vec);
    single_model_descriptor->input_nodes        = realloc_tensor_list(single_model_descriptor->input_nodes, single_model_descriptor->input_nodes_num);

    if (0 < single_model_descriptor->input_nodes_num &&
        NULL == single_model_descriptor->input_nodes) {
        err_print("construct nef single model information inputs tensor in model_descriptor fail: alloc memory fail ...\n");
        status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
        goto FUNC_OUT;
    }

    for (int idx = 0; idx < single_model_descriptor->input_nodes_num; idx++) {
        tensor                      = KneronKNE_Tensor_vec_at(tensor_vec, idx);
        tensor_descriptor           = &(single_model_descriptor->input_nodes[idx]);
        tensor_descriptor->index    = idx;

        status = construct_kne_single_model_tensors_info(tensor, single_model_descriptor->target, tensor_descriptor);
        if (KP_SUCCESS != status) {
            err_print("construct nef single model information inputs tensor in model_descriptor fail: construct tensor fail ...\n");
            goto FUNC_OUT;
        }
    }

FUNC_OUT:
    return status;
}

int construct_kne_single_model_output_tensor_info(KneronKNE_ModelHeader_table_t model_header, kp_single_model_descriptor_t *single_model_descriptor)
{
    int status                                  = KP_SUCCESS;
    kp_tensor_descriptor_t *tensor_descriptor   = NULL;
    KneronKNE_Tensor_vec_t tensor_vec;
    KneronKNE_Tensor_table_t tensor;

    if ((NULL == model_header) ||
        (NULL == single_model_descriptor)) {
        err_print("construct nef single model information outputs tensor in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    tensor_vec = KneronKNE_ModelHeader_outputs(model_header);
    if (NULL == tensor_vec) {
        err_print("construct nef single model information outputs tensor in model_descriptor fail: invalid flatbuffer ...\n");
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    single_model_descriptor->output_nodes_num   = KneronKNE_Tensor_vec_len(tensor_vec);
    single_model_descriptor->output_nodes       = realloc_tensor_list(single_model_descriptor->output_nodes, single_model_descriptor->output_nodes_num);

    if (0 < single_model_descriptor->output_nodes_num &&
        NULL == single_model_descriptor->output_nodes) {
        err_print("construct nef single model information outputs tensor in model_descriptor fail: alloc memory fail ...\n");
        status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
        goto FUNC_OUT;
    }

    for (int idx = 0; idx < single_model_descriptor->output_nodes_num; idx++) {
        tensor                      = KneronKNE_Tensor_vec_at(tensor_vec, idx);
        tensor_descriptor           = &(single_model_descriptor->output_nodes[idx]);
        tensor_descriptor->index    = idx;

        status = construct_kne_single_model_tensors_info(tensor, single_model_descriptor->target, tensor_descriptor);
        if (KP_SUCCESS != status) {
            err_print("construct nef single model information outputs tensor in model_descriptor fail: construct tensor fail ...\n");
            goto FUNC_OUT;
        }
    }

FUNC_OUT:
    return status;
}

int construct_kne_single_model_schema_info(KneronKNE_ModelHeader_table_t model_header, kp_single_model_descriptor_t *single_model_descriptor)
{
    int status = KP_SUCCESS;
    KneronKNE_SchemaVersion_table_t schema_version = NULL;

    if ((NULL == model_header) ||
        (NULL == single_model_descriptor)) {
        err_print("construct nef single model schema information in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    schema_version = KneronKNE_ModelHeader_schema_version(model_header);
    if (NULL == schema_version) {
        err_print("construct nef single model schema information in model_descriptor fail: invalid flatbuffer ...\n");
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    single_model_descriptor->setup_bin_schema_version.major     = KneronKNE_SchemaVersion_major_num(schema_version);
    single_model_descriptor->setup_bin_schema_version.minor     = KneronKNE_SchemaVersion_minor_num(schema_version);
    single_model_descriptor->setup_bin_schema_version.revision  = KneronKNE_SchemaVersion_revision_num(schema_version);

FUNC_OUT:
    return status;
}

int construct_kne_single_model_info(KneronKNE_Model_table_t kne_model, kp_single_model_descriptor_t *single_model_descriptor)
{
    int status = KP_SUCCESS;
    KneronKNE_ModelHeader_table_t model_header = NULL;

    if ((NULL == kne_model) ||
        (NULL == single_model_descriptor)) {
        err_print("construct nef single model information in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    model_header = KneronKNE_Model_header(kne_model);
    if (NULL == model_header) {
        err_print("construct nef single model information in model_descriptor fail: invalid flatbuffer ...\n");
        status = KP_ERROR_INVALID_MODEL_21;
        goto FUNC_OUT;
    }

    single_model_descriptor->id = KneronKNE_ModelHeader_id(model_header);

    status = construct_kne_single_model_schema_info(model_header, single_model_descriptor);
    if (KP_SUCCESS != status)
        goto FUNC_OUT;

    status = construct_kne_single_model_input_tensor_info(model_header, single_model_descriptor);
    if (KP_SUCCESS != status)
        goto FUNC_OUT;

    status = construct_kne_single_model_output_tensor_info(model_header, single_model_descriptor);
    if (KP_SUCCESS != status)
        goto FUNC_OUT;

FUNC_OUT:
    return status;
}

/******************************************************************
 * [private] KNE reader utils
 ******************************************************************/

int get_kne_header(KneronKNE_KNEContent_table_t* table_p, kp_kne_info_t *kne_info)
{
    if (table_p == NULL) {
        return KP_ERROR_INVALID_PARAM_12;
    }

    KneronKNE_KNEHeader_table_t nef_header = KneronKNE_KNEContent_header(*table_p);
    if (NULL == nef_header || NULL == kne_info) {
        return KP_ERROR_INVALID_MODEL_21;
    }

    kne_info->target        = KneronKNE_KNEHeader_target(nef_header);
    kne_info->kne_header    = (uintptr_t)nef_header;

    return KP_SUCCESS;
}

int get_kne_models(KneronKNE_KNEContent_table_t* table_p, kp_kne_info_t *kne_info)
{
    if (table_p == NULL) {
        return KP_ERROR_INVALID_PARAM_12;
    }

    KneronKNE_Model_vec_t kne_model_vec = KneronKNE_KNEContent_models(*table_p);
    if (NULL == kne_model_vec || NULL == kne_info) {
        return KP_ERROR_INVALID_MODEL_21;
    }

    kne_info->kne_model_vec = (uintptr_t)kne_model_vec;

    return KP_SUCCESS;
}

/******************************************************************
 * [public] KNE reader
 ******************************************************************/

int read_kne(uintptr_t kne_data, kp_kne_info_t *kne_info)
{
    int status                          = KP_SUCCESS;
    KneronKNE_KNEContent_table_t table  = KneronKNE_KNEContent_as_root((char*)kne_data);
    if (table == NULL)
        return KP_ERROR_INVALID_MODEL_21;

    status = get_kne_header(&table, kne_info);
    if (status != 0)
        return KP_ERROR_INVALID_MODEL_21;

    status = get_kne_models(&table, kne_info);
    if (status != 0)
        return KP_ERROR_INVALID_MODEL_21;

    return status;
}

int get_kne_single_model_output_buffer_size(uintptr_t kne_model_vec_ptr, uint32_t model_id, size_t *output_buffer_size)
{
    int status = KP_SUCCESS;
    KneronKNE_Model_vec_t kne_model_vec;
    KneronKNE_Model_table_t kne_model;
    KneronKNE_ModelHeader_table_t kne_model_header;
    KneronKNE_BufferInfo_vec_t buffer_info_vec;
    KneronKNE_BufferInfo_table_t buffer_info;
    if ((NULL == (void *)kne_model_vec_ptr) ||
        (NULL == output_buffer_size)) {
        err_print("get kne single model output buffer size fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    kne_model_vec = (KneronKNE_Model_vec_t)kne_model_vec_ptr;
    for (int model_idx = 0; model_idx < KneronKNE_Model_vec_len(kne_model_vec); model_idx++) {
        kne_model = KneronKNE_Model_vec_at(kne_model_vec, model_idx);
        if (NULL == kne_model) {
            err_print("get kne single model output buffer size fail: invalid flatbuffer ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }

        kne_model_header = KneronKNE_Model_header(kne_model);
        if (NULL == kne_model_header) {
            err_print("get kne single model output buffer size fail: invalid flatbuffer ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }

        if (model_id == KneronKNE_ModelHeader_id(kne_model_header)) {
            buffer_info_vec = KneronKNE_ModelHeader_buffer_info(kne_model_header);
            if (NULL == buffer_info_vec) {
                err_print("get kne single model output buffer size fail: invalid flatbuffer ...\n");
                status = KP_ERROR_INVALID_MODEL_21;
                goto FUNC_OUT;
            }

            for (int buff_info_idx = 0; buff_info_idx < KneronKNE_BufferInfo_vec_len(buffer_info_vec); buff_info_idx++) {
                buffer_info = KneronKNE_BufferInfo_vec_at(buffer_info_vec, buff_info_idx);
                if (NULL == buffer_info) {
                    err_print("get kne single model output buffer size fail: invalid flatbuffer ...\n");
                    status = KP_ERROR_INVALID_MODEL_21;
                    goto FUNC_OUT;
                }

                if (KneronKNE_Location_OUTPUT_BUFFER == KneronKNE_BufferInfo_buffer(buffer_info)) {
                    *output_buffer_size = KneronKNE_BufferInfo_len(buffer_info);
                    goto FUNC_OUT;
                }
            }
        }
    }

    status = KP_ERROR_MODEL_NOT_LOADED_35;

FUNC_OUT:
    return status;
}

int construct_kne_models_info(uintptr_t kne_model_vec_ptr, kp_model_nef_descriptor_t* loaded_model_desc)
{
    int status = KP_SUCCESS;
    KneronKNE_Model_vec_t kne_model_vec;
    if ((NULL == (void *)kne_model_vec_ptr) ||
        (NULL == loaded_model_desc)) {
        err_print("construct nef models information in model_descriptor fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    kne_model_vec                   = (KneronKNE_Model_vec_t)kne_model_vec_ptr;
    loaded_model_desc->num_models   = KneronKNE_Model_vec_len(kne_model_vec);
    loaded_model_desc->models       = realloc_model_descriptor_list(loaded_model_desc->models, loaded_model_desc->num_models);

    if ((0 < loaded_model_desc->num_models) &&
        (NULL == loaded_model_desc->models)) {
        err_print("construct nef models model_descriptor fail: realloc single model descriptor fail ...\n");
        status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
        goto FUNC_OUT;
    }

    for (int idx = 0; idx < loaded_model_desc->num_models; idx++) {
        KneronKNE_Model_table_t kne_model                       = KneronKNE_Model_vec_at(kne_model_vec, idx);
        kp_single_model_descriptor_t *single_model_descriptor   = &(loaded_model_desc->models[idx]);

        /* setting target chip from NEF metadata */
        single_model_descriptor->target                         = loaded_model_desc->target;

        status = construct_kne_single_model_info(kne_model, single_model_descriptor);
        if (KP_SUCCESS != status) {
            status = KP_ERROR_INVALID_PARAM_12;
            goto FUNC_OUT;
        }
    }

FUNC_OUT:
    return status;
}
