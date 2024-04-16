/*
 * This file holds USB constants and structures that are needed for
 * USB device APIs.  These are used by the USB device model, which is
 * defined in chapter 9 of the USB 2.0 specification and in the
 * Wireless USB 1.0 (spread around).
 *
 * USB 2.0 adds an additional "On The Go" (OTG) mode, which lets systems
 * act either as a USB master/host or as a USB slave/device.  That means
 * the master and slave side APIs benefit from working well together.
 *
 * There's also "Wireless USB", using low power short range radios for
 * peripheral interconnection but otherwise building on the USB framework.
 *
 * Note all descriptors are declared '__attribute__((packed))' so that:
 *
 * [a] they never get padded, either internally (USB spec writers
 *     probably handled that) or externally;
 *
 * [b] so that accessing bigger-than-a-bytes fields will never
 *     generate bus errors on any platform, even when the location of
 *     its descriptor inside a bundle isn't "naturally aligned", and
 *
 * [c] for consistency, removing all doubt even when it appears to
 *     someone that the two other points are non-issues for that
 *     particular descriptor type.
 *
 *  Copyright (C) 2019 Kneron, Inc. All rights reserved.
 *
 *
 */

#ifndef __KDP_USB_CH9_H__
#define __KDP_USB_CH9_H__
#include <stdint.h>
/*-------------------------------------------------------------------------*/

/* CONTROL REQUEST SUPPORT */

/*
 * USB directions
 *
 * This bit flag is used in endpoint descriptors' bEndpointAddress field.
 * It's also one of three fields in control requests bRequestType.
 */
#define USB_DIR_OUT           0x0        /* to device */
#define USB_DIR_IN            0x80        /* to host */

/*
 * USB types, the second of three bRequestType fields
 */
#define USB_TYPE_MASK            (0x03 << 5)
#define USB_TYPE_STANDARD        (0x00 << 5)
#define USB_TYPE_CLASS           (0x01 << 5)
#define USB_TYPE_VENDOR          (0x02 << 5)
#define USB_TYPE_RESERVED        (0x03 << 5)

/*
 * USB recipients, the third of three bRequestType fields
 */
#define USB_RECIP_MASK            0x1f
#define USB_RECIP_DEVICE          0x00
#define USB_RECIP_INTERFACE       0x01
#define USB_RECIP_ENDPOINT        0x02
#define USB_RECIP_OTHER           0x03
/* From Wireless USB 1.0 */
#define USB_RECIP_PORT            0x04
#define USB_RECIP_RPIPE           0x05

/*
 * Standard requests, for the bRequest field of a SETUP packet.
 *
 * These are qualified by the bRequestType field, so that for example
 * TYPE_CLASS or TYPE_VENDOR specific feature flags could be retrieved
 * by a GET_STATUS request.
 */
#define USB_REQ_GET_STATUS        0x00
#define USB_REQ_CLEAR_FEATURE     0x01
#define USB_REQ_SET_FEATURE       0x03
#define USB_REQ_SET_ADDRESS       0x05
#define USB_REQ_GET_DESCRIPTOR    0x06
#define USB_REQ_SET_DESCRIPTOR    0x07
#define USB_REQ_GET_CONFIGURATION 0x08
#define USB_REQ_SET_CONFIGURATION 0x09
#define USB_REQ_GET_INTERFACE     0x0A
#define USB_REQ_SET_INTERFACE     0x0B
#define USB_REQ_SYNCH_FRAME       0x0C
#define USB_REQ_SET_SEL           0x30
#define USB_REQ_SET_ISOCH_DELAY   0x31

#define USB_REQ_SET_ENCRYPTION    0x0D    /* Wireless USB */
#define USB_REQ_GET_ENCRYPTION    0x0E
#define USB_REQ_RPIPE_ABORT       0x0E
#define USB_REQ_SET_HANDSHAKE     0x0F
#define USB_REQ_RPIPE_RESET       0x0F
#define USB_REQ_GET_HANDSHAKE     0x10
#define USB_REQ_SET_CONNECTION    0x11
#define USB_REQ_SET_SECURITY_DATA 0x12
#define USB_REQ_GET_SECURITY_DATA 0x13
#define USB_REQ_SET_WUSB_DATA     0x14
#define USB_REQ_LOOPBACK_DATA_WRITE  0x15
#define USB_REQ_LOOPBACK_DATA_READ   0x16
#define USB_REQ_SET_INTERFACE_DS     0x17

/* specific requests for USB Power Delivery */
#define USB_REQ_GET_PARTNER_PDO       20
#define USB_REQ_GET_BATTERY_STATUS    21
#define USB_REQ_SET_PDO               22
#define USB_REQ_GET_VDM               23
#define USB_REQ_SEND_VDM              24


/* HID requests */
#define USB_REQ_GET_REPORT          0x01
#define USB_REQ_GET_IDLE            0x02
#define USB_REQ_GET_PROTOCOL        0x03
#define USB_REQ_SET_REPORT          0x09
#define USB_REQ_SET_IDLE            0x0A
#define USB_REQ_SET_PROTOCOL        0x0B

/* The Link Power Management (LPM) ECN defines USB_REQ_TEST_AND_SET command,
 * used by hubs to put ports into a new L1 suspend state, except that it
 * forgot to define its number ...
 */

/*
 * USB feature flags are written using USB_REQ_{CLEAR,SET}_FEATURE, and
 * are read as a bit array returned by USB_REQ_GET_STATUS.  (So there
 * are at most sixteen features of each type.)  Hubs may also support a
 * new USB_REQ_TEST_AND_SET_FEATURE to put ports into L1 suspend.
 */
#define USB_DEVICE_SELF_POWERED     0    /* (read only) */
#define USB_DEVICE_REMOTE_WAKEUP    1    /* dev may initiate wakeup */
#define USB_DEVICE_TEST_MODE        2    /* (wired high speed only) */
#define USB_DEVICE_BATTERY          2    /* (wireless) */
#define USB_DEVICE_B_HNP_ENABLE     3    /* (otg) dev may initiate HNP */
#define USB_DEVICE_WUSB_DEVICE      3    /* (wireless)*/
#define USB_DEVICE_A_HNP_SUPPORT    4    /* (otg) RH port supports HNP */
#define USB_DEVICE_A_ALT_HNP_SUPPORT 5    /* (otg) other RH port does */
#define USB_DEVICE_DEBUG_MODE        6    /* (special devices only) */

/*
 * Test Mode Selectors
 * See USB 2.0 spec Table 9-7
 */
#define    TEST_J        1
#define    TEST_K        2
#define    TEST_SE0_NAK  3
#define    TEST_PACKET   4
#define    TEST_FORCE_EN 5

/*
 * New Feature Selectors as added by USB 3.0
 * See USB 3.0 spec Table 9-7
 */
#define USB_DEVICE_U1_ENABLE    48    /* dev may initiate U1 transition */
#define USB_DEVICE_U2_ENABLE    49    /* dev may initiate U2 transition */
#define USB_DEVICE_LTM_ENABLE   50    /* dev may send LTM */
#define USB_INTRF_FUNC_SUSPEND   0    /* function suspend */

#define USB_INTR_FUNC_SUSPEND_OPT_MASK    0xFF00
/*
 * Suspend Options, Table 9-8 USB 3.0 spec
 */
#define USB_INTRF_FUNC_SUSPEND_LP    (1 << (8 + 0))
#define USB_INTRF_FUNC_SUSPEND_RW    (1 << (8 + 1))

/*
 * Interface status, Figure 9-5 USB 3.0 spec
 */
#define USB_INTRF_STAT_FUNC_RW_CAP     1
#define USB_INTRF_STAT_FUNC_RW         2

#define USB_ENDPOINT_HALT        0    /* IN/OUT will STALL */

/* Bit array elements as returned by the USB_REQ_GET_STATUS request. */
#define USB_DEV_STAT_U1_ENABLED        2    /* transition into U1 state */
#define USB_DEV_STAT_U2_ENABLED        3    /* transition into U2 state */
#define USB_DEV_STAT_LTM_ENABLED       4    /* Latency tolerance messages */

/*
 * Feature selectors from Table 9-8 USB Power Delivery spec
 */
#define USB_DEVICE_BATTERY_WAKE_MASK 40
#define USB_DEVICE_OS_IS_PD_AWARE    41
#define USB_DEVICE_POLICY_MODE       42
#define USB_PORT_PR_SWAP             43
#define USB_PORT_GOTO_MIN            44
#define USB_PORT_RETURN_POWER        45
#define USB_PORT_ACCEPT_PD_REQUEST   46
#define USB_PORT_REJECT_PD_REQUEST   47
#define USB_PORT_PORT_PD_RESET       48
#define USB_PORT_C_PORT_PD_CHANGE    49
#define USB_PORT_CABLE_PD_RESET      50
#define USB_DEVICE_CHARGING_POLICY   54

/**
 * struct usb_ctrlrequest - SETUP data for a USB device control request
 * @bRequestType: matches the USB bmRequestType field
 * @bRequest: matches the USB bRequest field
 * @wValue: matches the USB wValue field (le16 byte order)
 * @wIndex: matches the USB wIndex field (le16 byte order)
 * @wLength: matches the USB wLength field (le16 byte order)
 *
 * This structure is used to send control requests to a USB device.  It matches
 * the different fields of the USB 2.0 Spec section 9.3, table 9-2.  See the
 * USB spec for a fuller description of the different fields, and what they are
 * used for.
 *
 * Note that the driver for any interface can issue control requests.
 * For most devices, interfaces don't coordinate with each other, so
 * such requests may be made at any time.
 */
struct usb_ctrlrequest {
    uint8_t bRequestType;
    uint8_t bRequest;
    uint16_t wValue;
    uint16_t wIndex;
    uint16_t wLength;
} __attribute__ ((packed));



/*
 * Descriptor types ... USB 2.0 spec table 9.5
 */
#define USB_DT_DEVICE            0x01
#define USB_DT_CONFIG            0x02
#define USB_DT_STRING            0x03
#define USB_DT_INTERFACE         0x04
#define USB_DT_ENDPOINT          0x05
#define USB_DT_DEVICE_QUALIFIER  0x06
#define USB_DT_OTHER_SPEED_CONFIG    0x07
#define USB_DT_INTERFACE_POWER       0x08
/* these are from a minor usb 2.0 revision (ECN) */
#define USB_DT_OTG            0x09
#define USB_DT_DEBUG          0x0a
#define USB_DT_INTERFACE_ASSOCIATION 0x0b
/* these are from the Wireless USB spec */
#define USB_DT_SECURITY               0x0c
#define USB_DT_KEY                    0x0d
#define USB_DT_ENCRYPTION_TYPE        0x0e
#define USB_DT_BOS                    0x0f
#define USB_DT_DEVICE_CAPABILITY      0x10
#define USB_DT_WIRELESS_ENDPOINT_COMP 0x11
#define USB_DT_WIRE_ADAPTER           0x21
#define USB_DT_RPIPE                  0x22
#define USB_DT_CS_RADIO_CONTROL       0x23
/* From the T10 UAS specification */
#define USB_DT_PIPE_USAGE             0x24
/* From the USB 3.0 spec */
#define USB_DT_SS_ENDPOINT_COMP       0x30
/* From the USB 3.1 spec */
#define USB_DT_SSP_ISOC_ENDPOINT_COMP 0x31

/* Conventional codes for class-specific descriptors.  The convention is
 * defined in the USB "Common Class" Spec (3.11).  Individual class specs
 * are authoritative for their usage, not the "common class" writeup.
 */
#define USB_DT_CS_DEVICE        (USB_TYPE_CLASS | USB_DT_DEVICE)
#define USB_DT_CS_CONFIG        (USB_TYPE_CLASS | USB_DT_CONFIG)
#define USB_DT_CS_STRING        (USB_TYPE_CLASS | USB_DT_STRING)
#define USB_DT_CS_INTERFACE     (USB_TYPE_CLASS | USB_DT_INTERFACE)
#define USB_DT_CS_ENDPOINT      (USB_TYPE_CLASS | USB_DT_ENDPOINT)

/* All standard descriptors have these 2 fields at the beginning */
struct usb_descriptor_header {
    uint8_t  bLength;
    uint8_t  bDescriptorType;
} __attribute__ ((packed));

/*-------------------------------------------------------------------------*/

/* USB_DT_DEVICE: Device descriptor */
struct usb_device_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t bcdUSB;
    uint8_t  bDeviceClass;
    uint8_t  bDeviceSubClass;
    uint8_t  bDeviceProtocol;
    uint8_t  bMaxPacketSize0;
    uint16_t idVendor;
    uint16_t idProduct;
    uint16_t bcdDevice;   // bcdDeviceReleaseNumber
    uint8_t  iManufacturer;
    uint8_t  iProduct;
    uint8_t  iSerialNumber;
    uint8_t  bNumConfigurations;
} __attribute__ ((packed));

#define USB_DT_DEVICE_SIZE        18

struct usb_device_qualifier_descriptor {
    uint8_t bLength;// Number Size of descriptor
    uint8_t bDescriptorType; // Constant Device Qualifier Type
    uint16_t bcdUSB; // BCD USB specification version number (e.g., 0200H for V2.00 )
    uint8_t bDeviceClass; // Class Class Code
    uint8_t bDeviceSubClass; // SubClass SubClass Code
    uint8_t bDeviceProtocol; // Protocol Protocol Code
    uint8_t bMaxPacketSize0; // Number Maximum packet size for other speed
    uint8_t bNumConfigurations; // Number Number of Other-speed Configurations
    uint8_t bReserved; // Zero Reserved for future use, must be zero
} __attribute__ ((packed));

struct usb_other_speed_configuration_descriptor {
    uint8_t bLength; //  Number Size of descriptor
    uint8_t bDescriptorType; // Constant Other_speed_Configuration Type
    uint16_t wTotalLength; //  Number Total length of data returned
    uint8_t bNumInterfaces; // Number Number of interfaces supported by this speed configuration
    uint8_t bConfigurationValue; // 1 Number Value to use to select configuration
    uint8_t iConfiguration; // Index Index of string descriptor
    uint8_t bmAttributes; //Bitmap Same as Configuration descriptor
    uint8_t bMaxPower; // mA Same as Configuration descriptor
} __attribute__ ((packed));

struct usb_otg_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bmAttributes;     /* support for HNP, SRP, etc */
} __attribute__ ((packed));

/* USB_DT_OTG (from OTG 2.0 supplement) */
struct usb_otg20_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bmAttributes;     /* support for HNP, SRP and ADP, etc */
    uint16_t bcdOTG;          /* OTG and EH supplement release number
                                 * in binary-coded decimal(i.e. 2.0 is 0200H)
                                 */
} __attribute__ ((packed));
/*
 * Device and/or Interface Class codes
 * as found in bDeviceClass or bInterfaceClass
 * and defined by www.usb.org documents
 */
#define USB_CLASS_PER_INTERFACE    0    /* for DeviceClass */
#define USB_CLASS_AUDIO            1
#define USB_CLASS_COMM             2
#define USB_CLASS_HID              3
#define USB_CLASS_PHYSICAL         5
#define USB_CLASS_STILL_IMAGE      6
#define USB_CLASS_PRINTER          7
#define USB_CLASS_MASS_STORAGE     8
#define USB_CLASS_HUB              9
#define USB_CLASS_CDC_DATA         0x0a
#define USB_CLASS_CSCID            0x0b    /* chip+ smart card */
#define USB_CLASS_CONTENT_SEC      0x0d    /* content security */
#define USB_CLASS_VIDEO            0x0e
#define USB_CLASS_WIRELESS_CONTROLLER    0xe0
#define USB_CLASS_MISC            0xef
#define USB_CLASS_APP_SPEC        0xfe
#define USB_CLASS_VENDOR_SPEC     0xff

#define USB_SUBCLASS_VENDOR_SPEC  0xff

/*-------------------------------------------------------------------------*/

/* USB_DT_CONFIG: Configuration descriptor information.
 *
 * USB_DT_OTHER_SPEED_CONFIG is the same descriptor, except that the
 * descriptor type is different.  Highspeed-capable devices can look
 * different depending on what speed they're currently running.  Only
 * devices with a USB_DT_DEVICE_QUALIFIER have any OTHER_SPEED_CONFIG
 * descriptors.
 */
struct usb_config_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t wTotalLength;
    uint8_t  bNumInterfaces;
    uint8_t  bConfigurationValue;
    uint8_t  iConfiguration;
    uint8_t  bmAttributes;
    uint8_t  bMaxPower;
} __attribute__ ((packed));

#define USB_DT_CONFIG_SIZE        9

/* from config descriptor bmAttributes */
#define USB_CONFIG_ATT_ONE        (1 << 7)    /* must be set */
#define USB_CONFIG_ATT_SELFPOWER  (1 << 6)    /* self powered */
#define USB_CONFIG_ATT_WAKEUP     (1 << 5)    /* can wakeup */
#define USB_CONFIG_ATT_BATTERY    (1 << 4)    /* battery powered */

/*-------------------------------------------------------------------------*/

/* USB_DT_STRING: String descriptor */
struct usb_string_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t wData[1];        /* UTF-16LE encoded */
} __attribute__ ((packed));

/* note that "string" zero is special, it holds language codes that
 * the device supports, not Unicode characters.
 */

/*-------------------------------------------------------------------------*/

/* USB_DT_INTERFACE: Interface descriptor */
struct usb_interface_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bInterfaceNumber;
    uint8_t  bAlternateSetting;
    uint8_t  bNumEndpoints;
    uint8_t  bInterfaceClass;
    uint8_t  bInterfaceSubClass;
    uint8_t  bInterfaceProtocol;
    uint8_t  iInterface;
} __attribute__ ((packed));

#define USB_DT_INTERFACE_SIZE        9

/*-------------------------------------------------------------------------*/

/* USB_DT_ENDPOINT: Endpoint descriptor */
struct usb_endpoint_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bEndpointAddress;
    uint8_t  bmAttributes;
    uint16_t wMaxPacketSize;
    uint8_t  bInterval;

    /* NOTE:  these two are _only_ in audio endpoints. */
    /* use USB_DT_ENDPOINT*_SIZE in bLength, not sizeof. */
    uint8_t  bRefresh;
    uint8_t  bSynchAddress;
} __attribute__ ((packed));

#define USB_DT_ENDPOINT_SIZE          7
#define USB_DT_ENDPOINT_AUDIO_SIZE    9    /* Audio extension */

/*
 * Endpoints
 */
#define USB_ENDPOINT_NUMBER_MASK    0x0f    /* in bEndpointAddress */
#define USB_ENDPOINT_DIR_MASK       0x80

#define USB_ENDPOINT_XFERTYPE_MASK    0x03    /* in bmAttributes */
#define USB_ENDPOINT_XFER_CONTROL     0
#define USB_ENDPOINT_XFER_ISOC        1
#define USB_ENDPOINT_XFER_BULK        2
#define USB_ENDPOINT_XFER_INT         3
#define USB_ENDPOINT_MAX_ADJUSTABLE   0x80

/* The USB 3.0 spec redefines bits 5:4 of bmAttributes as interrupt ep type. */
#define USB_ENDPOINT_INTRTYPE         0x30
#define USB_ENDPOINT_INTR_PERIODIC    (0 << 4)
#define USB_ENDPOINT_INTR_NOTIFICATION    (1 << 4)

//#define USB_ENDPOINT_SYNCTYPE        0x0c
//#define USB_ENDPOINT_SYNC_NONE       (0 << 2)
//#define USB_ENDPOINT_SYNC_ASYNC      (1 << 2)
//#define USB_ENDPOINT_SYNC_ADAPTIVE   (2 << 2)
//#define USB_ENDPOINT_SYNC_SYNC       (3 << 2)

//#define USB_ENDPOINT_USAGE_MASK        0x30
//#define USB_ENDPOINT_USAGE_DATA        0x00
//#define USB_ENDPOINT_USAGE_FEEDBACK    0x10
//#define USB_ENDPOINT_USAGE_IMPLICIT_FB    0x20    /* Implicit feedback Data endpoint */

/*-------------------------------------------------------------------------*/

/**
 * usb_endpoint_num - get the endpoint's number
 * @epd: endpoint to be checked
 *
 * Returns @epd's number: 0 to 15.
 */
static inline int usb_endpoint_num(const struct usb_endpoint_descriptor *epd)
{
    return epd->bEndpointAddress & USB_ENDPOINT_NUMBER_MASK;
}

/**
 * usb_endpoint_type - get the endpoint's transfer type
 * @epd: endpoint to be checked
 *
 * Returns one of USB_ENDPOINT_XFER_{CONTROL, ISOC, BULK, INT} according
 * to @epd's transfer type.
 */
static inline int usb_endpoint_type(const struct usb_endpoint_descriptor *epd)
{
    return epd->bmAttributes & USB_ENDPOINT_XFERTYPE_MASK;
}

/**
 * usb_endpoint_dir_in - check if the endpoint has IN direction
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type IN, otherwise it returns false.
 */
static inline int usb_endpoint_dir_in(const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bEndpointAddress & USB_ENDPOINT_DIR_MASK) == USB_DIR_IN);
}

/**
 * usb_endpoint_dir_out - check if the endpoint has OUT direction
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type OUT, otherwise it returns false.
 */
static inline int usb_endpoint_dir_out(
    const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bEndpointAddress & USB_ENDPOINT_DIR_MASK) == USB_DIR_OUT);
}

/**
 * usb_endpoint_xfer_bulk - check if the endpoint has bulk transfer type
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type bulk, otherwise it returns false.
 */
static inline int usb_endpoint_xfer_bulk(
    const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bmAttributes & USB_ENDPOINT_XFERTYPE_MASK) ==
            USB_ENDPOINT_XFER_BULK);
}

/**
 * usb_endpoint_xfer_control - check if the endpoint has control transfer type
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type control, otherwise it returns false.
 */
static inline int usb_endpoint_xfer_control(
    const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bmAttributes & USB_ENDPOINT_XFERTYPE_MASK) ==
            USB_ENDPOINT_XFER_CONTROL);
}

/**
 * usb_endpoint_xfer_int - check if the endpoint has interrupt transfer type
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type interrupt, otherwise it returns
 * false.
 */
static inline int usb_endpoint_xfer_int(
    const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bmAttributes & USB_ENDPOINT_XFERTYPE_MASK) ==
            USB_ENDPOINT_XFER_INT);
}

/**
 * usb_endpoint_xfer_isoc - check if the endpoint has isochronous transfer type
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint is of type isochronous, otherwise it returns
 * false.
 */
static inline int usb_endpoint_xfer_isoc(
    const struct usb_endpoint_descriptor *epd)
{
    return ((epd->bmAttributes & USB_ENDPOINT_XFERTYPE_MASK) ==
            USB_ENDPOINT_XFER_ISOC);
}

/**
 * usb_endpoint_is_bulk_in - check if the endpoint is bulk IN
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has bulk transfer type and IN direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_bulk_in(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_bulk(epd) && usb_endpoint_dir_in(epd);
}

/**
 * usb_endpoint_is_bulk_out - check if the endpoint is bulk OUT
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has bulk transfer type and OUT direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_bulk_out(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_bulk(epd) && usb_endpoint_dir_out(epd);
}

/**
 * usb_endpoint_is_int_in - check if the endpoint is interrupt IN
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has interrupt transfer type and IN direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_int_in(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_int(epd) && usb_endpoint_dir_in(epd);
}

/**
 * usb_endpoint_is_int_out - check if the endpoint is interrupt OUT
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has interrupt transfer type and OUT direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_int_out(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_int(epd) && usb_endpoint_dir_out(epd);
}

/**
 * usb_endpoint_is_isoc_in - check if the endpoint is isochronous IN
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has isochronous transfer type and IN direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_isoc_in(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_isoc(epd) && usb_endpoint_dir_in(epd);
}

/**
 * usb_endpoint_is_isoc_out - check if the endpoint is isochronous OUT
 * @epd: endpoint to be checked
 *
 * Returns true if the endpoint has isochronous transfer type and OUT direction,
 * otherwise it returns false.
 */
static inline int usb_endpoint_is_isoc_out(
    const struct usb_endpoint_descriptor *epd)
{
    return usb_endpoint_xfer_isoc(epd) && usb_endpoint_dir_out(epd);
}

/**
 * usb_endpoint_maxp - get endpoint's max packet size
 * @epd: endpoint to be checked
 *
 * Returns @epd's max packet
 */
static inline int usb_endpoint_maxp(const struct usb_endpoint_descriptor *epd)
{
    return epd->wMaxPacketSize;
}

static inline int usb_endpoint_interrupt_type(
    const struct usb_endpoint_descriptor *epd)
{
    return epd->bmAttributes & USB_ENDPOINT_INTRTYPE;
}

/*-------------------------------------------------------------------------*/

/* USB_DT_SSP_ISOC_ENDPOINT_COMP: SuperSpeedPlus Isochronous Endpoint Companion
 * descriptor
 */
struct usb_ssp_isoc_ep_comp_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;
    uint16_t wReseved;
    uint32_t dwBytesPerInterval;
} __attribute__ ((packed));

#define USB_DT_SSP_ISOC_EP_COMP_SIZE        8

/*-------------------------------------------------------------------------*/

/* USB_DT_SS_ENDPOINT_COMP: SuperSpeed Endpoint Companion descriptor */
struct usb_ss_ep_comp_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bMaxBurst;
    uint8_t  bmAttributes;
    uint16_t wBytesPerInterval;
} __attribute__ ((packed));

#define USB_DT_SS_EP_COMP_SIZE        6

/* Bits 4:0 of bmAttributes if this is a bulk endpoint */
static inline int
usb_ss_max_streams(const struct usb_ss_ep_comp_descriptor *comp)
{
    int  max_streams;

    if (!comp)
        return 0;

    max_streams = comp->bmAttributes & 0x1f;

    if (!max_streams)
        return 0;

    max_streams = 1 << max_streams;

    return max_streams;
}

/* Bits 1:0 of bmAttributes if this is an isoc endpoint */
#define USB_SS_MULT(p)            (1 + ((p) & 0x3))
/* Bit 7 of bmAttributes if a SSP isoc endpoint companion descriptor exists */
#define USB_SS_SSP_ISOC_COMP(p)        ((p) & (1 << 7))

/*-------------------------------------------------------------------------*/

/* USB_DT_DEVICE_QUALIFIER: Device Qualifier descriptor */
struct usb_qualifier_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t bcdUSB;
    uint8_t  bDeviceClass;
    uint8_t  bDeviceSubClass;
    uint8_t  bDeviceProtocol;
    uint8_t  bMaxPacketSize0;
    uint8_t  bNumConfigurations;
    uint8_t  bRESERVED;
} __attribute__ ((packed));




/* from usb_otg_descriptor.bmAttributes */
#define USB_OTG_SRP        (1 << 0)
#define USB_OTG_HNP        (1 << 1)    /* swap host/device roles */
#define USB_OTG_ADP        (1 << 2)    /* support ADP */

#define OTG_STS_SELECTOR    0xF000        /* OTG status selector */
/*-------------------------------------------------------------------------*/

/* USB_DT_DEBUG:  for special highspeed devices, replacing serial console */
struct usb_debug_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    /* bulk endpoints with 8 byte maxpacket */
    uint8_t  bDebugInEndpoint;
    uint8_t  bDebugOutEndpoint;
} __attribute__((packed));

/*-------------------------------------------------------------------------*/

/* USB_DT_INTERFACE_ASSOCIATION: groups interfaces */
struct usb_interface_assoc_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bFirstInterface;
    uint8_t  bInterfaceCount;
    uint8_t  bFunctionClass;
    uint8_t  bFunctionSubClass;
    uint8_t  bFunctionProtocol;
    uint8_t  iFunction;
} __attribute__ ((packed));


/*-------------------------------------------------------------------------*/

/* USB_DT_SECURITY:  group of wireless security descriptors, including
 * encryption types available for setting up a CC/association.
 */
struct usb_security_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t wTotalLength;
    uint8_t  bNumEncryptionTypes;
} __attribute__((packed));

/*-------------------------------------------------------------------------*/

/* USB_DT_KEY:  used with {GET,SET}_SECURITY_DATA; only public keys
 * may be retrieved.
 */
struct usb_key_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  tTKID[3];
    uint8_t  bReserved;
    uint8_t  bKeyData[0];
} __attribute__((packed));

/*-------------------------------------------------------------------------*/

/* USB_DT_ENCRYPTION_TYPE:  bundled in DT_SECURITY groups */
struct usb_encryption_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint8_t  bEncryptionType;
#define    USB_ENC_TYPE_UNSECURE        0
#define    USB_ENC_TYPE_WIRED        1    /* non-wireless mode */
#define    USB_ENC_TYPE_CCM_1        2    /* aes128/cbc session */
#define    USB_ENC_TYPE_RSA_1        3    /* rsa3072/sha1 auth */
    uint8_t  bEncryptionValue;        /* use in SET_ENCRYPTION */
    uint8_t  bAuthKeyIndex;
} __attribute__((packed));


/*-------------------------------------------------------------------------*/

/* USB_DT_BOS:  group of device-level capabilities */
struct usb_bos_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;

    uint16_t wTotalLength;
    uint8_t  bNumDeviceCaps;
} __attribute__((packed));

#define USB_DT_BOS_SIZE        5


/*-------------------------------------------------------------------------*/

/* USB 2.0 defines three speeds, here's how Linux identifies them */

enum usb_device_speed {
    USB_SPEED_UNKNOWN = 0,            /* enumerating */
    USB_SPEED_LOW1,
    USB_SPEED_FULL1,        /* usb 1.1 */
    USB_SPEED_HIGH1,                /* usb 2.0 */
    USB_SPEED_WIRELESS,            /* wireless (usb 2.5) */
    USB_SPEED_SUPER,            /* usb 3.0 */
    USB_SPEED_SUPER_PLUS,            /* usb 3.1 */
};


enum usb_device_state {
    /* NOTATTACHED isn't in the USB spec, and this state acts
     * the same as ATTACHED ... but it's clearer this way.
     */
    USB_STATE_NOTATTACHED = 0,

    /* chapter 9 and authentication (wireless) device states */
    USB_STATE_ATTACHED,
    USB_STATE_POWERED,            /* wired */
    USB_STATE_RECONNECTING,       /* auth */
    USB_STATE_UNAUTHENTICATED,    /* auth */
    USB_STATE_DEFAULT,            /* limited function */
    USB_STATE_ADDRESS,
    USB_STATE_CONFIGURED,         /* most functions */

    USB_STATE_SUSPENDED

    /* NOTE:  there are actually four different SUSPENDED
     * states, returning to POWERED, DEFAULT, ADDRESS, or
     * CONFIGURED respectively when SOF tokens flow again.
     * At this level there's no difference between L1 and L2
     * suspend states.  (L2 being original USB 1.1 suspend.)
     */
};

enum usb3_link_state {
    USB3_LPM_U0 = 0,
    USB3_LPM_U1,
    USB3_LPM_U2,
    USB3_LPM_U3
};

/*
 * A U1 timeout of 0x0 means the parent hub will reject any transitions to U1.
 * 0xff means the parent hub will accept transitions to U1, but will not
 * initiate a transition.
 *
 * A U1 timeout of 0x1 to 0x7F also causes the hub to initiate a transition to
 * U1 after that many microseconds.  Timeouts of 0x80 to 0xFE are reserved
 * values.
 *
 * A U2 timeout of 0x0 means the parent hub will reject any transitions to U2.
 * 0xff means the parent hub will accept transitions to U2, but will not
 * initiate a transition.
 *
 * A U2 timeout of 0x1 to 0xFE also causes the hub to initiate a transition to
 * U2 after N*256 microseconds.  Therefore a U2 timeout value of 0x1 means a U2
 * idle timer of 256 microseconds, 0x2 means 512 microseconds, 0xFE means
 * 65.024ms.
 */
#define USB3_LPM_DISABLED        0x0
#define USB3_LPM_U1_MAX_TIMEOUT  0x7F
#define USB3_LPM_U2_MAX_TIMEOUT  0xFE
#define USB3_LPM_DEVICE_INITIATED    0xFF

struct usb_set_sel_req {
    uint8_t    u1_sel;
    uint8_t    u1_pel;
    uint16_t    u2_sel;
    uint16_t    u2_pel;
} __attribute__ ((packed));

/*
 * The Set System Exit Latency control transfer provides one byte each for
 * U1 SEL and U1 PEL, so the max exit latency is 0xFF.  U2 SEL and U2 PEL each
 * are two bytes long.
 */
#define USB3_LPM_MAX_U1_SEL_PEL        0xFF
#define USB3_LPM_MAX_U2_SEL_PEL        0xFFFF

/*-------------------------------------------------------------------------*/

/*
 * As per USB compliance update, a device that is actively drawing
 * more than 100mA from USB must report itself as bus-powered in
 * the GetStatus(DEVICE) call.
 * http://compliance.usb.org/index.asp?UpdateFile=Electrical&Format=Standard#34
 */
#define USB_SELF_POWER_VBUS_MAX_DRAW        100

#endif /* __KDP_USB_CH9_H__ */
