# ******************************************************************************
#  Copyright (c) 2022. Kneron Inc. All rights reserved.                        *
# ******************************************************************************

import os
import sys
import argparse

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))
sys.path.insert(1, os.path.join(PWD, '../example/'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
from utils.postprocess.bytetrack_postprocess import postprocess_ as bytetrack_postproc
from utils.postprocess.bytetrack_postprocess import BYTETracker

import kp
import cv2
import numpy as np
import math


def intv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def decode_outputs(outputs_np, hw_np, layer_strides=[8, 16, 32]):
    """decode outputs for a batch item into bbox predictions.
    Args:
        outputs_np: The anchor aggregated output, with last channel in reg(4), obj_score(1), cls_score(80), 4+1+80 = 85
                    [batch, n_anchors_all, 85] = [batch, 80x80 + 40x40 + 20x20, 85] = [1,8400,85]
        hw_np: the feature_size of the three PAN output layers, example: [(80, 80), (40, 40), (20, 20)] for 640x640
    Returns:
        The decoded: (Batch, anchor, 7): The last dimension is [cx, cy, w, h, obj_score, cls_score, label]
    """

    grids_np = []
    strides_np = []
    for (hsize, wsize), stride in zip(hw_np, layer_strides):
        '''
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        Giving the string ‘ij’ returns a meshgrid with matrix indexing,
        while ‘xy’ returns a meshgrid with Cartesian indexing.
        In the 2-D case with inputs of length M and N,
        the outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing
        '''
        yv, xv = np.meshgrid(np.arange(0, hsize), np.arange(0, wsize), indexing='ij')
        grid_np = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids_np.append(grid_np)
        shape = grid_np.shape[:2]
        strides_np.append(np.full((*shape, 1), stride, dtype=np.float32))

    grids_np = np.concatenate(grids_np, axis=1).astype(np.float32)
    strides_np = np.concatenate(strides_np, axis=1).astype(np.float32)

    outputs_np[..., :2] = (outputs_np[..., :2] + grids_np) * strides_np  # center
    outputs_np[..., 2:4] = np.exp(outputs_np[..., 2:4]) * strides_np  # bbox size

    return outputs_np


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold=0.5, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    indexes = np.argsort(scores)[::-1]

    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, 0)
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def yolox_postprocess(prediction_np, num_classes, prob_threshold=0.3, iou_threshold=0.5, topk=300):
    """
    Args:
        prediction_np: (Batch, anchor, 7): The last dimension is [cx, cy, w, h, obj_score, cls_score, label]
        num_classes: number of classes, for COCO = 80
        prob_threshold: threshold of scores filtering of object
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        box_probs: (N, 6): The detected objects, The last dimension is [x1, y1, x2, y2, score, label]
    """
    box_corner_np = np.zeros_like(prediction_np)
    box_corner_np[:, :, 0] = prediction_np[:, :, 0] - prediction_np[:, :, 2] / 2
    box_corner_np[:, :, 1] = prediction_np[:, :, 1] - prediction_np[:, :, 3] / 2
    box_corner_np[:, :, 2] = prediction_np[:, :, 0] + prediction_np[:, :, 2] / 2
    box_corner_np[:, :, 3] = prediction_np[:, :, 1] + prediction_np[:, :, 3] / 2
    prediction_np[:, :, :4] = box_corner_np[:, :, :4]

    # compute the argmax of all (COCO=80) classes and check if pass the threshold
    image_pred_np = prediction_np[0]
    class_conf_np = np.amax(image_pred_np[:, 5: 5 + num_classes], 1, keepdims=True)
    class_pred_np = np.argmax(image_pred_np[:, 5: 5 + num_classes], 1)
    class_pred_np = np.expand_dims(class_pred_np, -1)
    class_pred_np = class_pred_np.astype(np.float32)
    conf_mask_np = np.squeeze(image_pred_np[:, 4] * np.squeeze(class_conf_np) >= prob_threshold)

    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections_np = np.concatenate((image_pred_np[:, :5], class_conf_np, class_pred_np), 1)
    # x1 y1 x2 y2 obj_conf cls_conf, label
    detections_np = detections_np[conf_mask_np]

    box_probs = np.empty((0, 6), dtype=float)
    if not detections_np.shape[0]:
        return box_probs

    # class-wised NMS
    for class_index in range(num_classes):
        mask = np.squeeze(detections_np[:, 6] == class_index)
        cls_score = detections_np[:, 5]
        cls_score = cls_score[mask].reshape(-1, 1)
        if cls_score.shape[0] == 0:
            continue
        reg_bbox = detections_np[:, :4]
        reg_bbox = reg_bbox[mask]
        obj_score = detections_np[:, 4]
        obj_score = obj_score[mask].reshape(-1, 1)
        box_probs_class = np.concatenate([reg_bbox, obj_score * cls_score], axis=1)

        box_probs_class = hard_nms(box_probs_class, iou_threshold, topk)
        label = np.ones((box_probs_class.shape[0], 1)) * (class_index)  # adding label to box_prob
        box_probs_class = np.concatenate([box_probs_class, label], axis=1)
        box_probs = np.concatenate([box_probs, box_probs_class], axis=0)
    return box_probs


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_bboxes(outputs_np, max_shape, num_classes=80, prob_threshold=0.3, iou_threshold=0.5, topk=300):
    """Transform network output for a batch into bbox predictions.
    Args:
        outputs_np : a list of three PAN output layers, of reg_out, obj_out, cls_out
                    for 640x640. outpus_npshape  is with shape [ (1, 80, 80, 4), (1, 80, 80, 1)  (1, 80, 80, 80) ],
                                                                 [(1, 40, 40, 4), (1, 40, 40, 1), (1, 40, 40, 80) ],
                                                                 [(1, 20, 20, 4), (1, 20, 20, 1), (1, 20, 20, 80) ]
        max_shape: H,W onnx input size
        num_classes: number of classes, for COCO = 80
        prob_threshold: threshold of scores filtering of object
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        dets: (N, 6): The detected objects, The last dimension is [x1, y1, x2, y2, score, label]

    """
    # for 640x640. outpus_np shape is with shape (1, 80, 80, 85), (1, 40, 40, 85),(1, 20, 20, 85)
    # for 488x800 and 1 class: (1, 56, 100, 6), (1, 28, 50, 6), (1, 14, 25, 6)
    outputs_np = [np.concatenate([reg_out, sigmoid(obj_out), sigmoid(cls_out)], axis=3) \
                  for reg_out, obj_out, cls_out in outputs_np]

    # for 640x640 hw_np is with shape[(80, 80), (40, 40), (20, 20)]
    # for 488x800 ahw_np is with shape [(56, 100), (28, 50), (14, 25)]
    hw_np = [x.shape[1:3] for x in outputs_np]
    # [batch, n_anchors_all, 85] = [batch, 80x80 + 40x40 + 20x20, 85] = [1,8400,85]
    # =============== aggregate anchor start =================================
    for i in range(len(outputs_np)):
        batch, feature_h, feature_w, channel = outputs_np[i].shape
        outputs_np[i] = outputs_np[i].reshape(batch, -1, channel)
    outputs_np = np.concatenate([outputs_np[0], outputs_np[1], outputs_np[2]], axis=1)
    # outputs_np = np.transpose(outputs_np, (0,2,1))
    # =============== aggregate anchor end =================================

    prediction_np = decode_outputs(outputs_np, hw_np)
    dets = yolox_postprocess(prediction_np, num_classes, prob_threshold, iou_threshold, topk)

    for det in dets:
        det[0] = np.clip(det[0], 0, max_shape[1] - 1)
        det[1] = np.clip(det[1], 0, max_shape[0] - 1)
        det[2] = np.clip(det[2], 0, max_shape[1] - 1)
        det[3] = np.clip(det[3], 0, max_shape[0] - 1)
    return dets


def postprocess_(inf_results, **kwargs):
    """
    Input:
        outputs_np : a list of three PAN output layers, of reg_out, obj_out, cls_out
                    for 640x640. outpus_npshape  is with shape [ (1, 80, 80, 4), (1, 80, 80, 1)  (1, 80, 80, 80) ,
                                                                 (1, 40, 40, 4), (1, 40, 40, 1), (1, 40, 40, 80) ,
                                                                 (1, 20, 20, 4), (1, 20, 20, 1), (1, 20, 20, 80) ]
        input_shape: onnx input shape
        num_classes: number of classes
        conf_thres: confidence threshold
        iou_thres: iou threshold

    """
    num_pan_layer = 3
    outputs_np = []
    for i in range(num_pan_layer):
        reg_out = np.asarray(inf_results[3 * i + 0])
        obj_out = np.asarray(inf_results[3 * i + 1])
        cls_out = np.asarray(inf_results[3 * i + 2])
        assert reg_out.shape[3] > obj_out.shape[
            3], f"Regression node channel length smaller than objectness node channel length(reg: {reg_out.shape[3]} v.s. obj: {obj_out.shape[3]}), which is not allow in this postprocess function. Make sure your input array is in correct oreder."
        outputs_np.append([reg_out, obj_out, cls_out])

    input_shape = (kwargs['input_shape'][0], kwargs['input_shape'][1])

    # get the bbox from anchor information
    box_probs = get_bboxes(outputs_np=outputs_np, max_shape=input_shape, num_classes=kwargs['num_classes'],
                           prob_threshold=kwargs['conf_threshold'], iou_threshold=kwargs['iou_threshold'],
                           topk=kwargs['top_k_num'])

    # scale back to original image
    for i in range(len(box_probs)):
        box_probs[i, 2:4] = box_probs[i, 2:4] - box_probs[i, :2]

    return box_probs.tolist()


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
    parser = argparse.ArgumentParser(description='KL720 Kneron Model Zoo Generic Data Inference Example. - Bytetrack.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    parser.add_argument('-vid',
                        '--video_path',
                        help='input video path',
                        default=os.path.join(PWD, '../../res/images/MOT16-03_trim.mp4'),
                        type=str)
    parser.add_argument('-nef',
                        '--nef_model_path',
                        help='input NEF model path',
                        default=os.path.join(PWD,
                                             '../../res/models/KL720/kn-model-zoo-mmtracking_bytetrack/example_bytetrack_720.nef'),
                        type=str)
    parser.add_argument('--disable_gui',
                        metavar='Disable OpenCV Show Window',
                        nargs='?',
                        const=True,
                        default=False,
                        type=bool)
    args = parser.parse_args()

    usb_port_id = args.port_id
    video_file_path = args.video_path
    nef_model_path = args.nef_model_path
    disable_gui = args.disable_gui

    assert args.video_path is not None, "need to set input video but got None"
    assert args.nef_model_path is not None, "need to set nef model path but got None"

    track_thresh = 0.6
    match_thresh = 0.9
    frame_rate = 30
    track_buffer = 120
    tracker = BYTETracker(track_thresh, match_thresh, frame_rate, track_buffer)

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
                                                            file_path=nef_model_path)
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    extract input radix from NEF
    """
    nef_radix = model_nef_descriptor.models[0].input_nodes[0].quantization_parameters.quantized_fixed_point_descriptor_list[0].radix  # only support single model NEF

    """
    prepare the video
    """
    print('[Read Video]')
    nef_model_width = model_nef_descriptor.models[0].input_nodes[0].shape_npu[3]
    nef_model_height = model_nef_descriptor.models[0].input_nodes[0].shape_npu[2]

    cap = cv2.VideoCapture(video_file_path)

    output_img_name = 'output_{}'.format(os.path.basename(video_file_path))
    print(' - Save the result video to \'{}\''.format(output_img_name))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_vid = cv2.VideoWriter(output_img_name, fourcc, 20, (nef_model_width, nef_model_height))

    """
    starting inference work
    """
    print('[Starting Inference Work]')
    print(' - ', end='')
    frame_num = 0
    while cap.isOpened():
        ret, img = cap.read()
        if ret is False:
            break

        frame_num += 1
        img = cv2.resize(img, (nef_model_width, nef_model_height), interpolation=cv2.INTER_AREA)
        img_height, img_width, img_channels = img.shape

        # to rgb
        img_input = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

        # this model trained with normalize method: (data - 128)/256 , 
        img_input = img_input / 256.
        img_input -= 0.5

        # toolchain calculate the radix value from input data (after normalization), and set it into NEF model.
        # NPU will divide input data "2^radix" automatically, so, we have to scaling the input data here due to this reason.
        img_input *= pow(2, nef_radix)

        # convert rgb to rgba and width align 4, due to npu requirement.
        img_buffer = convert_numpy_to_rgba_and_width_align_4(img_input)

        generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
            model_id=model_nef_descriptor.models[0].id,
            inference_number=0,
            input_node_data_list=[kp.GenericInputNodeData(buffer=img_buffer)]
        )

        try:
            kp.inference.generic_data_inference_send(device_group=device_group,
                                                     generic_inference_input_descriptor=generic_inference_input_descriptor)

            generic_raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        """
        retrieve inference node output 
        """
        print('[Retrieve Inference Node Output ]')
        inf_node_output_list = []
        for node_idx in range(generic_raw_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                             generic_raw_result=generic_raw_result,
                                                                                             channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
                                                                                             )
            inf_node_output_list.append(inference_float_node_output)

        print(' - Success')

        tmp = []
        for o_n in inf_node_output_list:
            o_array = o_n.ndarray.copy()
            o_array = o_array.transpose((0, 2, 3, 1))
            tmp.append(o_array)

        out_data = [tmp[3], tmp[6], tmp[0], tmp[4], tmp[7], tmp[1],
                    tmp[5], tmp[8], tmp[2]]

        kwargs = {'input_shape': [nef_model_height, nef_model_width], 'num_classes': 1, 'conf_threshold': 0.1,
                  'iou_threshold': 0.7, 'top_k_num': 300}
        det_res = postprocess_(out_data, **kwargs)

        track_res, person_ids = bytetrack_postproc(det_res, tracker, min_box_area=120)
        for idx, bbox in enumerate(track_res):
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]

            id = person_ids[idx]
            cv2.putText(img, str(id), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 200, 200), 5,
                        cv2.LINE_AA)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(img, "Press 'Q' key to quit", (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (105, 100, 250),
                        2, cv2.LINE_AA)

        print('[Result] - frame {}'.format(frame_num))
        print(track_res)

        out_vid.write(img)

        if not disable_gui:
            cv2.imshow('bytetrack result', img)
            if cv2.waitKey(1) == ord('q'):
                break

        print('.', end='', flush=True)
    print()

    out_vid.release()
    cap.release()

    if not disable_gui:
        cv2.destroyAllWindows()
