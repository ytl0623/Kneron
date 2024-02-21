# Getting Started

## Environment Setup

### Platform and OS Requirements:

| OS                       | Platform      | Python Version          |
|--------------------------|---------------|-------------------------|
| Windows 10               | x86_64 64-bit | 3.5-3.9 (x86_64 64-bit) |
| Ubuntu 18.04             | x86_64 64-bit | 3.5-3.9 (x86_64 64-bit) |

### Upgrade pip (pip version >= 21.X.X)

```bash
$ python -m pip install --upgrade pip
```

### General

* opencv-python (Install using `python -m pip install opencv-python`)
* numpy (Install using `python -m pip install numpy`)

### Ubuntu

```bash
# Please install python package by the following sequence
$ python -m pip install ./package/{platform}/KneronPLUS-X.X.X-py3-none-any.whl
$ python -m pip install ./package/{platform}/KneronPLUS_YOLO-X.X.X-py3-none-any.whl
```

> #### Ubuntu only step
> * Need to install libusb and config USB permission on Ubuntu 
>   ```bash
>   $ sudo bash install_libusb.sh
>   ```
> * Or add following lines in `/etc/udev/rules.d/10-local.rules`
>   ```text
>   KERNEL=="ttyUSB*",ATTRS{idVendor}=="067b",ATTRS{idProduct}=="2303",MODE="0777",SYMLINK+="kneron_uart"
>   KERNEL=="ttyUSB*",ATTRS{idVendor}=="1a86",ATTRS{idProduct}=="7523",MODE="0777",SYMLINK+="kneron_pwr"
>   SUBSYSTEM=="usb",ATTRS{idVendor}=="3231",ATTRS{idProduct}=="0100",MODE="0666"
>   SUBSYSTEM=="usb",ATTRS{idVendor}=="3231",ATTRS{idProduct}=="0200",MODE="0666"
>   SUBSYSTEM=="usb",ATTRS{idVendor}=="3231",ATTRS{idProduct}=="0720",MODE="0666"
>   ```
>   and restatr service by following command
>   ```bash
>   $ sudo udevadm trigger
>   ```

### Windows

```bash
# Please install python package by the following sequence
$ python -m pip install .\package\{platform}\KneronPLUS-X.X.X-py3-none-any.whl
$ python -m pip install .\package\{platform}\KneronPLUS_YOLO-X.X.X-py3-none-any.whl
```

```bash
(kneron) D:\ytl\CCU\Thesis\Kneron\AB062D4C_Object detection in KL720 (Small model)\app_object_detection_80_class_light_weight>python -m pip install ./package\windows\KneronPLUS-1.3.0-py3-none-any.whl
Processing d:\ytl\ccu\thesis\kneron\ab062d4c_object detection in kl720 (small model)\app_object_detection_80_class_light_weight\package\windows\kneronplus-1.3.0-py3-none-any.whl
Requirement already satisfied: numpy in d:\anaconda\envs\kneron\lib\site-packages (from KneronPLUS==1.3.0) (1.23.5)
KneronPLUS is already installed with the same version as the provided wheel. Use --force-reinstall to force an installation of the wheel.

(kneron) D:\ytl\CCU\Thesis\Kneron\AB062D4C_Object detection in KL720 (Small model)\app_object_detection_80_class_light_weight>python -m pip install ./package\windows\KneronPLUS_YOLO-1.3.0-py3-none-any.whl
Processing d:\ytl\ccu\thesis\kneron\ab062d4c_object detection in kl720 (small model)\app_object_detection_80_class_light_weight\package\windows\kneronplus_yolo-1.3.0-py3-none-any.whl
Requirement already satisfied: KneronPLUS>=1.2.0 in d:\anaconda\envs\kneron\lib\site-packages (from KneronPLUS-YOLO==1.3.0) (1.3.0)
Requirement already satisfied: numpy in d:\anaconda\envs\kneron\lib\site-packages (from KneronPLUS>=1.2.0->KneronPLUS-YOLO==1.3.0) (1.23.5)
Installing collected packages: KneronPLUS-YOLO
Successfully installed KneronPLUS-YOLO-1.3.0
```

## Quick Start

```bash
# Please check camera is connected firstly.
$ python cam_object_detection_80_class_light_weight.py
```

## Usage

```bash
$ python cam_object_detection_80_class_light_weight.py -h
```

* Show all input parameters for the APP.

```bash
$ python cam_object_detection_80_class_light_weight.py
```

* Run APP with first scanned kneron device.

```bash
$ python cam_object_detection_80_class_light_weight.py -p port_id
```

* Run APP with  specified port ID for connecting device (`Hint: find USB port ID by ScanDevice.py`).

```bash
$ python ScanDevice.py
```

* Show all connected Kneron AI Devices.

```bash
(kneron) D:\ytl\CCU\Thesis\Kneron\AB062D4C_Object detection in KL720 (Small model)\app_object_detection_80_class_light_weight>python ScanDevices.py
scanning kneron devices ...
number of Kneron devices found: 1
listing devices infomation as follows:

[0] USB scan index: '0'
[0] USB port ID: '73'
[0] Product ID: '0x720 (KL720)'
[0] USB link speed: '4'
[0] USB port path: '1-18'
[0] KN number: '0xAB062D4C'
[0] Connectable: 'True'
[0] Firmware: 'KDP2 Comp / F'
```

## Common Problem
* If pip install/run application fails, it may cause by using python 2.X as python interpreter. Please make sure the interpreter and pip is `Python 3` on the host:
    ```bash
    # check pip version
    $ pip -V
    $ pip3 -V

    # check python interpreter version
    $ python -V
    $ python3 -V
    ```

    You also can install package by specify python interpreter by following scripts:
    ```bash
    $ python -m pip install {package_path}
    # or
    $ python3 -m pip install {package_path}
    ```