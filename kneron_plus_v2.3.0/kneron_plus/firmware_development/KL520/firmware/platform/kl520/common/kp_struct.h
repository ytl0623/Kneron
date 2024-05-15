/**
 * @file        kp_struct.h
 * @brief       Kneron PLUS data structure
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#define APP_PADDING_BYTES 28                            /**< Default padding size */
#define MAX_INPUT_NODE_COUNT 5                          /**< Supported maximum count of the model input node */

#define KDP2_MAGIC_TYPE_COMMAND 0xAB67CD13              /**< Magic number for data check */
#define KDP2_MAGIC_TYPE_INFERENCE 0x11FF22AA            /**< Magic number for data check */
#define KDP2_MAGIC_TYPE_CUSTOMIZED 0x11FF33CC           /**< Magic number for customized data */
#define KDP2_MAGIC_TYPE_CHECKPOINT_DATA 0x34ABF977      /**< Magic number for debug checkpoint data */

#ifdef AUTOTEST /* CI_PACK_REMOVE_START */
#define KDP2_MAGIC_TYPE_JSON 0x22EE3A86                 /**< Magic number for json file stream */
#endif /* CI_PACK_REMOVE_END */

/**
 * @brief return code of most APIs.
 */
enum KP_API_RETURN_CODE
{
    KP_SUCCESS = 0,

    /* libusb error code */
    KP_ERROR_USB_IO_N1 = -1,
    KP_ERROR_USB_INVALID_PARAM_N2 = -2,
    KP_ERROR_USB_ACCESS_N3 = -3,
    KP_ERROR_USB_NO_DEVICE_N4 = -4,
    KP_ERROR_USB_NOT_FOUND_N5 = -5,
    KP_ERROR_USB_BUSY_N6 = -6,
    KP_ERROR_USB_TIMEOUT_N7 = -7,
    KP_ERROR_USB_OVERFLOW_N8 = -8,
    KP_ERROR_USB_PIPE_N9 = -9,
    KP_ERROR_USB_INTERRUPTED_N10 = -10,
    KP_ERROR_USB_NO_MEM_N11 = -11,
    KP_ERROR_USB_NOT_SUPPORTED_N12 = -12,
    KP_ERROR_USB_OTHER_N99 = -99,

    /* libwdi error code (remapping from: kneron_plus/thirdparty/windows/include/libwdi.h) */
    KP_ERROR_WDI_BEGIN = -200,
    KP_ERROR_WDI_IO_N1 = -201,
    KP_ERROR_WDI_INVALID_PARAM_N2 = -202,
    KP_ERROR_WDI_ACCESS_N3 = -203,
    KP_ERROR_WDI_NO_DEVICE_N4 = -204,
    KP_ERROR_WDI_NOT_FOUND_N5 = -205,
    KP_ERROR_WDI_BUSY_N6 = -206,
    KP_ERROR_WDI_TIMEOUT_N7 = -207,
    KP_ERROR_WDI_OVERFLOW_N8 = -208,
    KP_ERROR_WDI_PENDING_INSTALLATION_N9 = -209,
    KP_ERROR_WDI_INTERRUPTED_N10 = -210,
    KP_ERROR_WDI_RESOURCE_N11 = -211,
    KP_ERROR_WDI_NOT_SUPPORTED_N12 = -212,
    KP_ERROR_WDI_EXISTS_N13 = -213,
    KP_ERROR_WDI_USER_CANCEL_N14 = -214,
    KP_ERROR_WDI_NEEDS_ADMIN_N15 = -215,
    KP_ERROR_WDI_WOW64_N16 = -216,
    KP_ERROR_WDI_INF_SYNTAX_N17 = -217,
    KP_ERROR_WDI_CAT_MISSING_N18 = -218,
    KP_ERROR_WDI_UNSIGNED_N19 = -219,
    KP_ERROR_WDI_OTHER_N99 = -299,

    /* host error code */
    KP_ERROR_MEMORY_ALLOCATION_FAILURE_9 = 9,
    KP_ERROR_DEVICE_NOT_EXIST_10 = 10,
    KP_ERROR_DEVICE_INCORRECT_RESPONSE_11 = 11,
    KP_ERROR_INVALID_PARAM_12 = 12,
    KP_ERROR_SEND_DESC_FAIL_13 = 13,
    KP_ERROR_SEND_DATA_FAIL_14 = 14,
    KP_ERROR_SEND_DATA_TOO_LARGE_15 = 15,
    KP_ERROR_RECV_DESC_FAIL_16 = 16,
    KP_ERROR_RECV_DATA_FAIL_17 = 17,
    KP_ERROR_RECV_DATA_TOO_LARGE_18 = 18,
    KP_ERROR_FW_UPDATE_FAILED_19 = 19,
    KP_ERROR_FILE_OPEN_FAILED_20 = 20,
    KP_ERROR_INVALID_MODEL_21 = 21,
    KP_ERROR_IMAGE_RESOLUTION_TOO_SMALL_22 = 22,
    KP_ERROR_IMAGE_INVALID_WIDTH_23 = 23,
    KP_ERROR_INVALID_FIRMWARE_24 = 24,
    KP_ERROR_RESET_FAILED_25 = 25,
    KP_ERROR_DEVICES_NUMBER_26 = 26,
    KP_ERROR_CONFIGURE_DEVICE_27 = 27,
    KP_ERROR_CONNECT_FAILED_28 = 28,
    KP_ERROR_DEVICE_GROUP_MIX_PRODUCT_29 = 29,
    KP_ERROR_RECEIVE_INCORRECT_HEADER_STAMP_30 = 30,
    KP_ERROR_RECEIVE_SIZE_MISMATCH_31 = 31,
    KP_ERROR_RECEIVE_JOB_ID_MISMATCH_32 = 32,
    KP_ERROR_INVALID_CUSTOMIZED_JOB_ID_33 = 33,
    KP_ERROR_FW_LOAD_FAILED_34 = 34,
    KP_ERROR_MODEL_NOT_LOADED_35 = 35,
    KP_ERROR_INVALID_CHECKPOINT_DATA_36 = 36,
    KP_DBG_CHECKPOINT_END_37 = 37,
    KP_ERROR_INVALID_HOST_38 = 38,
    KP_ERROR_MEMORY_FREE_FAILURE_39 = 39,
    KP_ERROR_USB_BOOT_LOAD_SECOND_MODEL_40 = 40,
    KP_ERROR_CHECK_FW_VERSION_FAILED_41 = 41,
    KP_ERROR_FIFOQ_INPUT_BUFF_COUNT_NOT_ENOUGH_42 = 42,
    KP_ERROR_FIFOQ_SETTING_FAILED_43 = 43,
    KP_ERROR_UNSUPPORTED_DEVICE_44 = 44,

    KP_ERROR_OTHER_99 = 99,

    /* firmware error code */
    KP_FW_ERROR_UNKNOWN_APP = 100,
    KP_FW_INFERENCE_ERROR_101 = 101,
    KP_FW_DDR_MALLOC_FAILED_102 = 102,
    KP_FW_INFERENCE_TIMEOUT_103 = 103,
    KP_FW_LOAD_MODEL_FAILED_104 = 104,
    KP_FW_CONFIG_POST_PROC_ERROR_MALLOC_FAILED_105 = 105,
    KP_FW_CONFIG_POST_PROC_ERROR_NO_SPACE_106 = 106,
    KP_FW_IMAGE_SIZE_NOT_MATCH_MODEL_INPUT_107 = 107,
    KP_FW_NOT_SUPPORT_PREPROCESSING_108 = 108,
    KP_FW_GET_MODEL_INFO_FAILED_109 = 109,
    KP_FW_WRONG_INPUT_BUFFER_COUNT_110 = 110,
    KP_FW_INVALID_PRE_PROC_MODEL_INPUT_SIZE_111 = 111,
    KP_FW_INVALID_INPUT_CROP_PARAM_112 = 112,
    KP_FW_ERROR_FILE_OPEN_FAILED_113 = 113,
    KP_FW_ERROR_FILE_STATE_FAILED_114 = 114,
    KP_FW_ERROR_FILE_READ_FAILED_115 = 115,
    KP_FW_ERROR_FILE_WRITE_FAILED_116 = 116,
    KP_FW_ERROR_FILE_CHMOD_FAILED_117 = 117,
    KP_FW_ERROR_FILE_FAILED_OTHER_118 = 118,
    KP_FW_ERROR_INVALID_BOOT_CONFIG_119 = 119,
    KP_FW_ERROR_LOADER_ERROR_120 = 120,
    KP_FW_ERROR_POSIX_SPAWN_FAILED_121 = 121,
    KP_FW_ERROR_USB_SEND_FAILED_122 = 122,
    KP_FW_ERROR_USB_RECEIVE_FAILED_123 = 123,

    /* ncpu error code (sync with ipc.h) */
    KP_FW_NCPU_ERR_BEGIN         = 200,
    KP_FW_NCPU_INVALID_IMAGE_201 = 201,

    /* firmware eFuse error code */
    KP_FW_EFUSE_CAN_NOT_BURN_300 = 300,
    KP_FW_EFUSE_PROTECTED_301 = 301,
    KP_FW_EFUSE_OTHER_302 = 302,
};

/**
 * @brief enum for USB speed mode
 */
typedef enum
{
    KP_USB_SPEED_UNKNOWN = 0,
    KP_USB_SPEED_LOW = 1,
    KP_USB_SPEED_FULL = 2,
    KP_USB_SPEED_HIGH = 3,
    KP_USB_SPEED_SUPER = 4,
} kp_usb_speed_t;

/**
 * @brief enum for USB PID(Product ID)
 */
typedef enum
{
    KP_DEVICE_KL520 = 0x100,                /**< KL520 USB PID */
    KP_DEVICE_KL720 = 0x720,                /**< KL720 USB PID */
    KP_DEVICE_KL720_LEGACY = 0x200,         /**< KL720 Legacy USB PID */
    KP_DEVICE_KL530 = 0x530,                /**< KL530 USB PID */
    KP_DEVICE_KL730 = 0x730,                /**< KL730 USB PID */
    KP_DEVICE_KL630 = 0x630,                /**< KL630 USB PID */
    KP_DEVICE_KL540 = 0x540,                /**< KL540 USB PID */
} kp_product_id_t;

/**
 * @brief reset mode
 */
typedef enum
{
    KP_RESET_REBOOT = 0,        /**< Higheset level to reset Kneron device. Kneron device would disconnect after this reset. */
    KP_RESET_INFERENCE = 1,     /**< Soft reset: reset inference FIFO queue. */
    KP_RESET_SHUTDOWN = 2,      /**< Shut down Kneron device. For KL520, only useful if HW circuit supports (ex. 96 bord), dongle is not supported. For KL720, this function is not supported. */
    KP_RESET_REBOOT_SYSTEM = 3, /**< Reboot entire system */
} kp_reset_mode_t;

/**
 * @brief enum for generic raw data channel ordering
 */
typedef enum
{
    KP_CHANNEL_ORDERING_HCW = 0,            /**< KL520 default,     height/channel/width in order */
    KP_CHANNEL_ORDERING_CHW = 1,            /**< KL720 default,     channel/height/width in order */
    KP_CHANNEL_ORDERING_HWC = 2,            /**< TensorFlow style,  height/width/channel in order */
} kp_channel_ordering_t;

/**
 * @brief enum for fixed-point data type
 */
typedef enum
{
    KP_FIXED_POINT_DTYPE_UNKNOWN = 0,       /**< unknown data type */
    KP_FIXED_POINT_DTYPE_INT8 = 1,          /**< represent one fixed-point value by 8-bit data type */
    KP_FIXED_POINT_DTYPE_INT16 = 2,         /**< represent one fixed-point value by 16-bit data type */
} kp_fixed_point_dtype_t;

/**
 * @brief information of device (USB)
 */
typedef struct
{
    uint32_t port_id;                       /**< an unique ID representing for a Kneron device, can be used as input while connecting devices */
    uint16_t vendor_id;                     /**< supposed to be 0x3231. */
    uint16_t product_id;                    /**< enum kp_product_id_t. */
    int link_speed;                         /**< enum kp_usb_speed_t. */
    uint32_t kn_number;                     /**< KN number. */
    bool isConnectable;                     /**< indicate if this device is connectable. */
    char port_path[20];                     /**< "busNo-hub_portNo-device_portNo"
                                                        ex: "1-2-3", means bus 1 - (hub) port 2 - (device) port 3 */
    char firmware[30];                      /**< Firmware description. */
} __attribute__((aligned(4))) kp_device_descriptor_t;

/**
 * @brief information of connected devices from USB perspectives.
 */
typedef struct
{
    int num_dev;                            /**< connected devices */
    kp_device_descriptor_t device[];        /**< real index range from 0 ~ (num_dev-1) */
} __attribute__((aligned(4))) kp_devices_list_t;

/**
 * @brief image format supported for inference.
 */
typedef enum
{
    KP_IMAGE_FORMAT_UNKNOWN = 0x0,
    KP_IMAGE_FORMAT_RGB565 = 0x60,               /**< RGB565 16bits */
    KP_IMAGE_FORMAT_RGBA8888 = 0x0D,             /**< RGBA8888 32bits */
    KP_IMAGE_FORMAT_YUYV = 0x2F,                 /**< YUYV 16bits */
    KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0 = 0x30,    /**< YCbCr422 (order: CrY1CbY0) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0 = 0x31,    /**< YCbCr422 (order: CbY1CrY0) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB = 0x32,    /**< YCbCr422 (order: Y1CrY0Cb) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR = 0x33,    /**< YCbCr422 (order: Y1CbY0Cr) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1 = 0x34,    /**< YCbCr422 (order: CrY0CbY1) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1 = 0x35,    /**< YCbCr422 (order: CbY0CrY1) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB = 0x36,    /**< YCbCr422 (order: Y0CrY1Cb) 16bits */
    KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR = 0x37,    /**< YCbCr422 (order: Y0CbY1Cr) 16bits */
    KP_IMAGE_FORMAT_RAW8 = 0x20,                 /**< RAW 8bits */
} kp_image_format_t;

/**
 * @brief npu raw data layout format for tensors.
 */
typedef enum
{
    KP_MODEL_TENSOR_DATA_LAYOUT_UNKNOWN = 0,
    KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B = 1,     /**< width: 4  bits, channel: 4  bits, depth: 8  bits */
    KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B = 2,    /**< width: 1  bits, channel: 16 bits, depth: 8  bits */
    KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B = 3,    /**< width: 16 bits, channel: 4  bits, depth: 8  bits */
    KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B = 4,    /**< width: 8  bits, channel: 1  bits, depth: 16 bits */
} kp_model_tensor_data_layout_t;

/**
 * @brief model target chip.
 */
typedef enum
{
    KP_MODEL_TARGET_CHIP_UNKNOWN = 0,
    KP_MODEL_TARGET_CHIP_KL520 = 1,     /**< model for kl520 */
    KP_MODEL_TARGET_CHIP_KL720 = 2,     /**< model for kl720 */
    KP_MODEL_TARGET_CHIP_KL530 = 3,     /**< model for kl530 */
    KP_MODEL_TARGET_CHIP_KL730 = 4,     /**< model for kl730 */
    KP_MODEL_TARGET_CHIP_KL630 = 5,     /**< model for kl630 */
    KP_MODEL_TARGET_CHIP_KL540 = 6,     /**< model for kl540 */
} kp_model_target_chip_t;

/**
 * @brief a basic descriptor for nef schema version
 */
typedef struct
{
    uint32_t major;     /**< major number */
    uint32_t minor;     /**< minor number */
    uint32_t revision;  /**< revision number */
} __attribute__((packed, aligned(4))) kp_nef_schema_version_t;

/**
 * @brief a basic descriptor for setup.bin schema version
 */
typedef struct
{
    uint32_t major;     /**< major number */
    uint32_t minor;     /**< minor number */
    uint32_t revision;  /**< revision number */
} __attribute__((packed, aligned(4))) kp_setup_bin_schema_version_t;

/**
 * @brief a basic descriptor for setup.bin file schema version
 */
typedef struct
{
    uint32_t major;     /**< major number */
    uint32_t minor;     /**< minor number */
    uint32_t revision;  /**< revision number */
} __attribute__((packed, aligned(4))) kp_file_schema_version_t;

/**
 * @brief a basic descriptor for a fixed-point quantization information
 */
typedef struct
{
    float       scale;  /**< scale of node */
    int32_t     radix;  /**< radix of node */
} __attribute__((packed, aligned(4))) kp_quantized_fixed_point_descriptor_t;

/**
 * @brief a basic descriptor for quantization parameters
 */
typedef struct
{
    uint32_t                                quantized_fixed_point_descriptor_num;   /**< numbers of fixed-point quantization information */
    kp_quantized_fixed_point_descriptor_t*  quantized_fixed_point_descriptor;       /**< array of fixed-point quantization information */
} __attribute__((packed, aligned(4))) kp_quantization_parameters_t;

/**
 * @brief a basic descriptor for a node in model
 */
typedef struct
{
    uint32_t                        index;                      /**< index of node */
    char*                           name;                       /**< name of node */
    uint32_t                        shape_npu_len;              /**< length of npu shape */
    uint32_t*                       shape_npu;                  /**< npu shape */
    uint32_t                        shape_onnx_len;             /**< length of onnx shape */
    uint32_t*                       shape_onnx;                 /**< onnx shape */
    uint32_t                        data_layout;                /**< npu memory layout */
    kp_quantization_parameters_t    quantization_parameters;    /**< quantization parameters */
} __attribute__((packed, aligned(4))) kp_tensor_descriptor_t;

/**
 * @brief a basic descriptor for a model
 */
typedef struct
{
    uint32_t                        target;                     /**< target chip of model */
    uint32_t                        version;                    /**< version of model */
    uint32_t                        id;                         /**< id of model */
    uint32_t                        input_nodes_num;            /**< number of model input nodes */
    kp_tensor_descriptor_t*         input_nodes;                /**< array of model output node information */
    uint32_t                        output_nodes_num;           /**< number of model output nodes */
    kp_tensor_descriptor_t*         output_nodes;               /**< array of model output node information */
    kp_setup_bin_schema_version_t   setup_bin_schema_version;   /**< schema version of setup.bin */
    kp_file_schema_version_t        file_schema_version;        /**< file schema version of setup.bin */
    uint32_t                        max_raw_out_size;           /**< needed raw output buffer size for this model */
} __attribute__((packed, aligned(4))) kp_single_model_descriptor_t;

/**
 * @brief a basic descriptor for a NEF metadata
 */
typedef struct
{
    uint32_t                            kn_num;                 /**< target KN number device of encrypted all models */
    char*                               toolchain_version;      /**< toolchain version of all models */
    char*                               compiler_version;       /**< compiler version of all models */
    kp_nef_schema_version_t             nef_schema_version;     /**< schema version of nef */
    char*                               platform;               /**< usb dongle, 96 board, etc. */
} __attribute__((packed, aligned(4))) kp_model_nef_metadata_t;

/**
 * @brief a basic descriptor for a NEF
 */
typedef struct
{
    uint32_t                            magic;                  /**< magic number for model_nef_descriptor (0x5AA55AA5) */
    kp_model_nef_metadata_t             metadata;               /**< nef metadata */
    uint32_t                            target;                 /**< target chip of all models (1: KL520, 2: KL720, etc.) */
    uint32_t                            crc;                    /**< crc of all models */
    uint32_t                            num_models;             /**< number of models */
    kp_single_model_descriptor_t*       models;                 /**< model descriptors */
} __attribute__((packed, aligned(4))) kp_model_nef_descriptor_t;

/**
 * @brief attribute for configuring ddr
 */
typedef struct
{
    uint32_t model_size;                    /**< DDR space for model */
    uint32_t input_buffer_size;             /**< input buffer size for FIFO queue */
    uint32_t input_buffer_count;            /**< input buffer count for FIFO queue */
    uint32_t result_buffer_size;            /**< result buffer size for FIFO queue */
    uint32_t result_buffer_count;           /**< result buffer count for FIFO queue */
} __attribute__((aligned(4))) kp_ddr_manage_attr_t;

/**
 * @brief a handle represent connected Kneron device.
 */
typedef struct
{
    int timeout;                                    /**< global timeout value for all USB communications with the device */
    int num_device;                                 /**< number of devices in device group */
    kp_product_id_t product_id;                     /**< enum kp_product_id_t */
    kp_model_nef_descriptor_t loaded_model_desc;    /**< a basic descriptor for a NEF */
    kp_ddr_manage_attr_t ddr_attr;                  /**< attribute for configuring ddr */
} __attribute__((aligned(4))) kp_device_group_s;

/**
 * @brief a pointer handle represent connected Kneron device.
 */
typedef kp_device_group_s *kp_device_group_t;

/**
 * @brief normalization mode
 */
typedef enum
{
    KP_NORMALIZE_DISABLE = 0xFF,               /**< disable normalize */
    KP_NORMALIZE_KNERON = 0x1,                 /**< RGB/256 - 0.5, refer to the toolchain manual */
    KP_NORMALIZE_TENSOR_FLOW = 0x2,            /**< RGB/127.5 - 1.0, refer to the toolchain manual */
    KP_NORMALIZE_YOLO = 0x3,                   /**< RGB/255.0, refer to the toolchain manual */
    KP_NORMALIZE_CUSTOMIZED_DEFAULT = 0x4,     /**< customized, default, refer to the toolchain manual */
    KP_NORMALIZE_CUSTOMIZED_SUB128 = 0x5,      /**< customized, subtract 128, refer to the toolchain manual */
    KP_NORMALIZE_CUSTOMIZED_DIV2 = 0x6,        /**< customized, divide by 2, refer to the toolchain manual */
    KP_NORMALIZE_CUSTOMIZED_SUB128_DIV2 = 0x7, /**< customized, subtract 128 and divide by 2, refer to the toolchain manual */
} kp_normalize_mode_t;

/**
 * @brief resize mode
 */
typedef enum
{
    KP_RESIZE_DISABLE = 0x1,    /**< Disable Resize in Pre-process */
    KP_RESIZE_ENABLE = 0x2,     /**< Enable Resize in Pre-process */
} kp_resize_mode_t;

/**
 * @brief padding mode
 */
typedef enum
{
    KP_PADDING_DISABLE = 0x1,   /**< Disable Padding in Pre-process */
    KP_PADDING_CORNER = 0x2,    /**< Using Corner Padding in Pre-process */
    KP_PADDING_SYMMETRIC = 0x3, /**< Using Symmetric Padding in Pre-process */
} kp_padding_mode_t;

/**
 * @brief data structure for inference configurations
 */
typedef struct
{
    bool enable_frame_drop;                 /**< enable this to keep inference non-blocking by dropping oldest and unprocessed frames */
} __attribute__((aligned(4))) kp_inf_configuration_t;

/**
 * @brief data structure for a crop
 */
typedef struct
{
    uint32_t crop_number;                   /**< index number */

    uint32_t x1;                            /**< top-left corner: x */
    uint32_t y1;                            /**< top-left corner: y */
    uint32_t width;                         /**< width */
    uint32_t height;                        /**< height */
} __attribute__((packed, aligned(4))) kp_inf_crop_box_t;

/**
 * @brief hardware pre-process related value for raw output result
 *
 */
typedef struct
{
    uint32_t img_width;                     /**< image width before hardware pre-process */
    uint32_t img_height;                    /**< image height before hardware pre-process */
    uint32_t resized_img_width;             /**< image width after resize */
    uint32_t resized_img_height;            /**< image height after resize */
    uint32_t pad_top;                       /**< pixels padding on top */
    uint32_t pad_bottom;                    /**< pixels padding on bottom */
    uint32_t pad_left;                      /**< pixels padding on left */
    uint32_t pad_right;                     /**< pixels padding on right */
    uint32_t model_input_width;             /**< model required input width */
    uint32_t model_input_height;            /**< model required input height */
    kp_inf_crop_box_t crop_area;            /**< info of crop area (may not be the same as input due to hw limit) */
} __attribute__((packed, aligned(4))) kp_hw_pre_proc_info_t;

#define MAX_CROP_BOX 4 /**< MAX crop count */

/**
 * @brief inference RAW descriptor for one image
 */
typedef struct
{
    uint32_t width;                             /**< image width */
    uint32_t height;                            /**< image height */
    uint32_t resize_mode;                       /**< resize mode, refer to kp_resize_mode_t */
    uint32_t padding_mode;                      /**< padding mode, refer to kp_resize_mode_t */
    uint32_t image_format;                      /**< image format, refer to kp_image_format_t */
    uint32_t normalize_mode;                    /**< inference normalization, refer to kp_normalize_mode_t */
    uint32_t crop_count;                        /**< crop count */
    kp_inf_crop_box_t inf_crop[MAX_CROP_BOX];   /**< box information to crop */
    uint8_t *image_buffer;                      /**< image buffer */
} __attribute__((packed, aligned(4))) kp_generic_input_node_image_t;

/**
 * @brief inference RAW descriptor for one image under bypass pre process
 *
 */
typedef struct
{
    uint32_t buffer_size;                       /**< buffer size */
    uint8_t *buffer;                            /**< buffer of input data*/
} __attribute__((packed, aligned(4))) kp_generic_input_node_data_t;

/**
 * @brief inference descriptor for images
 */
typedef struct
{
    uint32_t inference_number;                                                  /**< inference sequence number */
    uint32_t model_id;                                                          /**< target inference model ID */
    uint32_t num_input_node_image;                                              /**< number of images for input nodes */
    kp_generic_input_node_image_t input_node_image_list[MAX_INPUT_NODE_COUNT];  /**< list of image data for each input node(maps to input nodes order of model) */
} __attribute__((packed, aligned(4))) kp_generic_image_inference_desc_t;

/**
 * @brief inference RAW output descriptor
 */
typedef struct
{
    uint32_t inference_number;                                          /**< inference sequence number */
    uint32_t crop_number;                                               /**< crop box sequence number */
    uint32_t num_output_node;                                           /**< total number of output nodes */
    uint32_t product_id;                                                /**< product id, refer to kp_product_id_t */
    uint32_t num_pre_proc_info;                                         /**< number of pre_proc_info is available */
    kp_hw_pre_proc_info_t pre_proc_info[MAX_INPUT_NODE_COUNT];          /**< hardware pre-process related value */
} __attribute__((packed, aligned(4))) kp_generic_image_inference_result_header_t;

/**
 * @brief inference descriptor for multiple input images bypass pre-processing
 */
typedef struct
{
    uint32_t inference_number;                                                  /**< inference sequence number */
    uint32_t model_id;                                                          /**< target inference model ID */
    uint32_t num_input_node_data;                                               /**< number of data for input nodes */
    kp_generic_input_node_data_t input_node_data_list[MAX_INPUT_NODE_COUNT];    /**< list of data for each input node(maps to input nodes order of model) */
} __attribute__((packed, aligned(4))) kp_generic_data_inference_desc_t;

/**
 * @brief inference RAW output descriptor for multiple input and bypass pre-processing
 */
typedef struct
{
    uint32_t inference_number;              /**< inference sequence number */
    uint32_t crop_number;                   /**< crop box sequence number */
    uint32_t num_output_node;               /**< total number of output nodes */
    uint32_t product_id;                    /**< product id, refer to kp_product_id_t */
} __attribute__((packed, aligned(4))) kp_generic_data_inference_result_header_t;

/**
 * @brief Metadata of RAW node output in fixed-point format
 */
typedef struct
{
    uint32_t height;                        /**< node height */
    uint32_t channel;                       /**< node channel */
    uint32_t width;                         /**< node width, should be aligned to 16 bytes for futher processing due to low level output */
    int32_t radix;                          /**< radix for fixed/floating point conversion */
    float scale;                            /**< scale for fixed/floating point conversion */
    uint32_t data_layout;                   /**< npu memory layout (ref. kp_model_tensor_data_layout_t) */
} __attribute__((aligned(4))) kp_inf_raw_fixed_node_metadata_t;

/**
 * @brief RAW node output in raw fixed-point format (with width padding and device channel ordering)
 */
typedef struct
{
    kp_inf_raw_fixed_node_metadata_t metadata;  /**< metadata of RAW node output in fixed-point format */
    uint32_t num_data;                          /**< total number of fixed-poiont values, should be
                                                     metadata->width (aligned to 16 bytes) * metadata->height * metadata->channel */
    int8_t *data;                               /**< array of fixed-point values*/
} __attribute__((aligned(4))) kp_inf_raw_fixed_node_output_t;

/**
 * @brief data of fixed-point values in 8-bits/16-bits (depended on fixed_point_dtype)
 */
typedef union
{
    int8_t int8[1];                     /**< array of fixed-point values in 8-bits */
    int16_t int16[1];                   /**< array of fixed-point values in 16-bits */
} fixed_node_output_data_t;

/**
 * @brief RAW node output in fixed-point format
 */
typedef struct
{
    uint32_t width;                         /**< node width */
    uint32_t height;                        /**< node height */
    uint32_t channel;                       /**< node channel */
    int32_t radix;                          /**< radix for fixed/floating point conversion */
    float scale;                            /**< scale for fixed/floating point conversion */
    float factor;                           /**< conversion factor for fixed-point to floating-point conversion - formula: 1 / (scale * (2 ^ radix)) */
    uint32_t fixed_point_dtype;             /**< enum kp_fixed_point_dtype_t */
    uint32_t num_data;                      /**< total number of fixed-point values */
    fixed_node_output_data_t data;          /**< data of fixed-point values in 8-bits/16-bits (depended on fixed_point_dtype) ref. fixed_node_output_data_t */
} __attribute__((aligned(4))) kp_inf_fixed_node_output_t;

/**
 * @brief RAW node output in floating-point format
 */
typedef struct
{
    uint32_t width;                         /**< node width */
    uint32_t height;                        /**< node height */
    uint32_t channel;                       /**< node channel */
    uint32_t num_data;                      /**< total number of floating-point values */
    float data[];                           /**< array of floating-point values */
} __attribute__((aligned(4))) kp_inf_float_node_output_t;

/**
 * @brief describe a bounding box
 */
typedef struct
{
    float x1;                               /**< top-left corner: x */
    float y1;                               /**< top-left corner: y */
    float x2;                               /**< bottom-right corner: x */
    float y2;                               /**< bottom-right corner: y */
    float score;                            /**< probability score */
    int32_t class_num;                      /**< class # (of many) with highest probability */
} __attribute__((aligned(4))) kp_bounding_box_t;

#define YOLO_GOOD_BOX_MAX 100 /**< maximum number of bounding boxes for Yolo models */

/**
 * @brief describe a yolo output result after post-processing
 */
typedef struct
{
    uint32_t class_count;                   /**< total class count detectable by model */
    uint32_t box_count;                     /**< boxes of all classes */
    kp_bounding_box_t boxes[YOLO_GOOD_BOX_MAX]; /**< box information */
} __attribute__((aligned(4))) kp_yolo_result_t;

#define LAND_MARK_POINTS 5 /**< the number of land marks points */

/**
 * @brief decribe a point
 */
typedef struct
{
    uint32_t x;                             /**< x value */
    uint32_t y;                             /**< y value */
} __attribute__((aligned(4))) kp_point_t;

/**
 * @brief describe a landmark
 */
typedef struct
{
    kp_point_t marks[LAND_MARK_POINTS];     /**< landmark points */
    float score;                            /**< score of this landmark */
    float blur;                             /**< blur score of this landmark */
    int32_t class_num;                      /**< class number */
} __attribute__((aligned(4))) kp_landmark_result_t;

#define FR_FEAT_LENGTH 256 /**< the length of one feature map */

/**
 * @brief describe a feature map
 */
typedef struct {
    float feature_map[FR_FEAT_LENGTH];          /**< feature map in floating point */
    int8_t feature_map_fixed[FR_FEAT_LENGTH];   /**< feature map in fixed point */
} __attribute__((aligned(4))) kp_fr_result_t;

/**
 * @brief describe a classification result
 */
typedef struct
{
    int32_t class_num;  /**< class # (of many) with highest probability */
    float   score;      /**< probability score */
} __attribute__((aligned(4))) kp_classification_result_t;

/**
 * @brief describe version string
 */
typedef struct
{
    uint8_t reserved;                       /**< for backward compatibility */
    uint8_t major;                          /**< major number */
    uint8_t minor;                          /**< minor number */
    uint8_t update;                         /**< update number */
    uint32_t build;                         /**< build number */
} __attribute__((aligned(4))) kp_firmware_version_t;

/**
 * @brief describe system information
 */
typedef struct
{
    uint32_t kn_number;                     /**< Chip K/N number */
    kp_firmware_version_t firmware_version; /**< FW version */
} __attribute__((aligned(4))) kp_system_info_t;

/**
 * @brief header stamp for user-defined data transfer
 */
typedef struct
{
    uint32_t magic_type;                    /**< must be 'KDP2_MAGIC_TYPE_XXXXXX' */
    uint32_t total_size;                    /**< total size of user-defined header data struct and data (image) */
    uint32_t job_id;                        /**< user-defined ID to synchronize with firmware side, must >= 1000 */
    uint32_t status_code;                   /**< this field is valid only for result data, refer to KP_API_RETURN_CODE */
    uint32_t total_image;                   /**< total number of images for this inference */
    uint32_t image_index;                   /**< the index of the image in this transmission */
} __attribute__((aligned(4))) kp_inference_header_stamp_t;

/**
 * @brief Inference debug checkpoints in bit-fields format
 */
typedef enum
{
    KP_DBG_CHECKPOINT_BEFORE_PREPROCESS = 0x1 << 0,    /**< Checkpoint data(image) at before-pre_processing stage */
    KP_DBG_CHECKPOINT_AFTER_PREPROCESS  = 0x1 << 1,    /**< Checkpoint data(image) at after-pre_processing stage */
    KP_DBG_CHECKPOINT_AFTER_INFERENCE   = 0x1 << 2,    /**< Checkpoint data(fixed-point raw) at after-inference stage */
    KP_DBG_CHECKPOINT_BEFORE_CPU_OP     = 0x1 << 3,    /**< Checkpoint data(cpu operation) at before-cpu operation stage */
    KP_DBG_CHECKPOINT_AFTER_CPU_OP      = 0x1 << 4,    /**< Checkpoint data(cpu operation) at after-cpu operation stage */
} kp_dbg_checkpoint_flag_t;

/**
 * @brief Inference debug data structure represents for "before-pre_process"
 */
typedef struct
{
    kp_inference_header_stamp_t header_stamp; /**< magic_type = 'KDP2_MAGIC_TYPE_CHECKPOINT_DATA' */
    uint32_t checkpoint_tag;                  /**< refer to kp_dbg_checkpoint_flag_t */
    uint32_t img_x;                           /**< image position X */
    uint32_t img_y;                           /**< image position Y */
    uint32_t img_width;                       /**< image width in pixels */
    uint32_t img_height;                      /**< image height in pixels */
    uint32_t img_format;                      /**< image format, refer to kp_image_format_t */
    int target_inf_model;                     /**< inferencing model */
    uint32_t img_index;                       /**< index of input image */
    uint8_t image[];                          /**< image raw data */
} __attribute__((aligned(4))) kp_dbg_checkpoint_data_before_preprocess_t;

/**
 * @brief Inference debug data structure represents for "after-pre_process"
 */
typedef struct
{
    kp_inference_header_stamp_t header_stamp; /**< magic_type = 'KDP2_MAGIC_TYPE_CHECKPOINT_DATA' */
    uint32_t checkpoint_tag;                  /**< refer to kp_dbg_checkpoint_flag_t */
    uint32_t img_width;                       /**< image width in pixels */
    uint32_t img_height;                      /**< image height in pixels */
    uint32_t img_format;                      /**< image format, refer to kp_image_format_t */
    int target_inf_model;                     /**< inferencing model */
    uint32_t img_index;                       /**< index of input image */
    uint8_t image[];                          /**< image raw data */
} __attribute__((aligned(4))) kp_dbg_checkpoint_data_after_preprocess_t;

/**
 * @brief Inference debug data structure represents for "after-inference"
 */
typedef struct
{
    kp_inference_header_stamp_t header_stamp;           /**< magic_type = 'KDP2_MAGIC_TYPE_CHECKPOINT_DATA' */
    uint32_t checkpoint_tag;                            /**< refer to kp_dbg_checkpoint_flag_t */
    int target_inf_model;                               /**< inferencing model */
    uint32_t num_nodes;                                 /**< number of output nodes */
    kp_inf_raw_fixed_node_metadata_t node_metadata[50]; /**< output node metada */
    uint32_t total_output_size;                         /**< total raw output size in bytes */
    uint8_t raw_output[];                               /**< truly raw output from NPU */
} __attribute__((aligned(4))) kp_dbg_checkpoint_data_after_inference_t;

/**
 * @brief Inference debug data structure represents for "before-cpu operation"
 */
typedef struct
{
    kp_inference_header_stamp_t header_stamp;           /**< magic_type = 'KDP2_MAGIC_TYPE_CHECKPOINT_DATA' */
    uint32_t checkpoint_tag;                            /**< refer to kp_dbg_checkpoint_flag_t */
    int target_inf_model;                               /**< inferencing model */
    uint32_t num_nodes;                                 /**< number of output nodes */
    kp_inf_raw_fixed_node_metadata_t node_metadata[50]; /**< output node metada */
    uint32_t total_output_size;                         /**< total raw output size in bytes */
    uint8_t raw_output[];                               /**< truly raw output from NPU */
} __attribute__((aligned(4))) kp_dbg_checkpoint_data_before_cpu_op_t;

/**
 * @brief Inference debug data structure represents for "after-cpu operation"
 */
typedef struct
{
    kp_inference_header_stamp_t header_stamp;           /**< magic_type = 'KDP2_MAGIC_TYPE_CHECKPOINT_DATA' */
    uint32_t checkpoint_tag;                            /**< refer to kp_dbg_checkpoint_flag_t */
    int target_inf_model;                               /**< inferencing model */
    uint32_t num_nodes;                                 /**< number of output nodes */
    kp_inf_raw_fixed_node_metadata_t node_metadata[50]; /**< output node metada */
    uint32_t total_output_size;                         /**< total raw output size in bytes */
    uint8_t raw_output[];                               /**< truly raw output from NPU */
} __attribute__((aligned(4))) kp_dbg_checkpoint_data_after_cpu_op_t;

typedef struct
{
    uint32_t model_id;                  /**< model ID */
    uint32_t inf_count;                 /**< number of inference  */
    uint32_t cpu_op_count;              /**< number of cpu operation per inference  */
    float avg_pre_process_ms;           /**< average pre-process time in milliseconds  */
    float avg_inference_ms;             /**< average inference time in milliseconds  */
    float avg_cpu_op_ms;                /**< average cpu operation time per-inference in milliseconds  */
    float avg_cpu_op_per_cpu_node_ms;   /**< average cpu operation time per-cpu node in milliseconds  */
    float avg_post_process_ms;          /**< average post-process time in milliseconds  */
} kp_profile_model_statistics_t;

typedef struct
{
    int num_model_profiled;                 /**< number of models profiled */
    kp_profile_model_statistics_t model_st[16]; /**< refer to kp_profile_model_statistics_t */
} __attribute__((aligned(4))) kp_profile_data_t;

/**
 * @brief Describe DDR memory space current configuration
 */
typedef struct
{
    uint32_t ddr_available_begin;       /**< Available DDR space begin address */
    uint32_t ddr_available_end;         /**< Available DDR space end address */
    uint32_t ddr_model_end;             /**< Model used DDR space end address */
    uint32_t ddr_fifoq_allocated;       /**< Whether FIFO queue has been configured */
} __attribute__((aligned(4))) kp_available_ddr_config_t;

/**
 * @brief Describe FIFO Queue current configuration
 */
typedef struct
{
    uint32_t fifoq_input_buf_count;     /**< Input buffer count for FIFO queue, 0 if FIFO queue has not been set */
    uint32_t fifoq_input_buf_size;      /**< Input buffer size for FIFO queue, 0 if FIFO queue has not been set */
    uint32_t fifoq_result_buf_count;    /**< Input buffer count for FIFO queue, 0 if FIFO queue has not been set */
    uint32_t fifoq_result_buf_size;     /**< Input buffer size for FIFO queue, 0 if FIFO queue has not been set */
} __attribute__((aligned(4))) kp_fifo_queue_config_t;
