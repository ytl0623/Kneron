import kp

USB_SPEED = 'usb_speed'
PRODUCT_ID = 'product_id'
DEVICE_RULE_DICT = {
    'KL520': {
        USB_SPEED: kp.UsbSpeed.KP_USB_SPEED_HIGH,
        PRODUCT_ID: kp.ProductId.KP_DEVICE_KL520.value
    },
    'KL720': {
        USB_SPEED: kp.UsbSpeed.KP_USB_SPEED_SUPER,
        PRODUCT_ID: kp.ProductId.KP_DEVICE_KL720.value
    }
}


def get_device_usb_speed_by_port_id(usb_port_id: int) -> kp.UsbSpeed:
    device_list = kp.core.scan_devices()

    for device_descriptor in device_list.device_descriptor_list:
        if 0 == usb_port_id:
            return device_descriptor.link_speed
        elif usb_port_id == device_descriptor.usb_port_id:
            return device_descriptor.link_speed

    raise IOError('Specified USB port ID {} not exist.'.format(usb_port_id))


def get_product_id_by_port_id(usb_port_id: int) -> int:
    device_list = kp.core.scan_devices()

    for device_descriptor in device_list.device_descriptor_list:
        if 0 == usb_port_id:
            return device_descriptor.product_id
        elif usb_port_id == device_descriptor.usb_port_id:
            return device_descriptor.product_id

    raise IOError('Specified USB port ID {} not exist.'.format(usb_port_id))


def check_device(usb_port_id: int, device_type: str):
    print('[Check Device]')

    global USB_SPEED, PRODUCT_ID, DEVICE_RULE_DICT

    if device_type not in DEVICE_RULE_DICT.keys():
        print('Error: unsupported device type = \'{}\', please make sure your config.json is correct.'.format(device_type))
        exit(0)

    # check device is for this application
    try:
        if DEVICE_RULE_DICT[device_type][PRODUCT_ID] != get_product_id_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Error] Device is not supported on this application.' + '\033[0m')
            exit(0)
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)

    # check device USB speed (Recommend run KL520 at high speed)
    try:
        if DEVICE_RULE_DICT[device_type][USB_SPEED] != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
            print('\033[91m' + '[Error] Device is not run at high speed.' + '\033[0m')
            exit(0)
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)
