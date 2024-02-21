import cv2
import numpy as np
from typing import List


def load_logo(logo_png_path, logo_resize_ratio: float = 0.6):
    logo_image = cv2.imread(logo_png_path, cv2.IMREAD_UNCHANGED)
    logo_image = cv2.resize(src=logo_image, dsize=(
        int(logo_image.shape[1] * logo_resize_ratio), int(logo_image.shape[0] * logo_resize_ratio)))

    # prepare LOGO image
    logo_image_mask = np.zeros(shape=[logo_image.shape[0], logo_image.shape[1], 3])
    logo_image_mask[logo_image[:, :, 3] != 0] = True

    logo_image = cv2.cvtColor(src=logo_image, code=cv2.COLOR_BGRA2BGR)

    return logo_image, logo_image_mask


def render_logo(image, logo_image, logo_image_mask):
    image[:logo_image.shape[0], image.shape[1] - logo_image.shape[1] - 10: -10][logo_image_mask == True] = logo_image[
        logo_image_mask == True]


def render_menu(image, menu_list: List[str]):
    x = 10
    y = image.shape[0] - 10
    line_height = 23
    font_scale = 1.5
    color = (200, 200, 200)
    font = cv2.FONT_HERSHEY_PLAIN

    menu_list = menu_list.copy()
    menu_list.reverse()

    for idx, menu in enumerate(menu_list):
        cv2.putText(img=image,
                    text=' - ' + menu,
                    org=(x, y - (line_height * idx)),
                    fontFace=font,
                    fontScale=font_scale,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

    cv2.putText(img=image,
                text='Function Keys',
                org=(x, y - (line_height * len(menu_list))),
                fontFace=font,
                fontScale=font_scale,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)


def render_fps(image, fps: float, org=(10, 30)):
    # draw FPS
    cv2.putText(img=image,
                text='FPS: {:.2f}'.format(fps),
                org=org,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(200, 200, 200),
                thickness=1,
                lineType=cv2.LINE_AA)


def render_box(image, x1, y1, x2, y2, color=(255, 150, 70), text=None):
    cv2.rectangle(img=image,
                  pt1=(int(x1), int(y1)),
                  pt2=(int(x2), int(y2)),
                  color=color,
                  thickness=3)

    if text is not None:
        cv2.putText(img=image,
                    text=text,
                    org=(int(x1), int(y1) - 8),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)


def render_text(image, text, x1, y1, color=(255, 150, 70)):
    cv2.putText(img=image,
                text=text,
                org=(int(x1), int(y1)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)
