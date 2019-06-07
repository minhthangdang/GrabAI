import cv2


# resize an image to a certain width and keep ratio
def resize(image, width=800):
    (h, w) = image.shape[:2]
    ratio = width / float(w)
    height = int(h * ratio)
    ret_img = cv2.resize(image, (width, height))
    return ret_img, ratio
