import cv2
import numpy as np

def is_skin(r, g, b, h, s, y, cr, cb):
    """
    判斷一個像素是否為膚色。
    """
    rgb_condition = (
        r > 100 and g > 45 and b > 30 and r > g and r > b and abs(r - g) > 15
    )
    hsv_condition = s > 35
    ycbcr_condition = (
        cr > 145
        and cb > 75
        and y > 80
        and cr <= (1.9 * cb) + 40
        and cr >= (0.2448 * cb) + 7.209
        and cr >= (-4.5652 * cb) + 234.5652
        and cr <= (-1.25 * cb) + 301.75
        and cr <= (-2.2857 * cb) + 432.85
    )

    return rgb_condition and hsv_condition and ycbcr_condition


def skin_detect(image_path, ground_truth_path):
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.GaussianBlur(image, (3, 3), 0)

    height, width, channels = image.shape

    skin_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv[0], hsv[1], hsv[2]

            ycbcr = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2YCrCb)[0][0]
            yc, cr, cb = ycbcr[0], ycbcr[1], ycbcr[2]
            
            if is_skin(r, g, b, h, s, yc, cr, cb):
                skin_mask[y, x] = 255
            else:
                skin_mask[y, x] = 0
    
    # 後處理
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2) # 開運算
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # 閉運算
        
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
            
    iou = calculate_iou(skin_mask, ground_truth)

    return skin_mask, iou

def calculate_iou(mask1, mask2):
    """計算兩個二值化遮罩的 IOU"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    
    return iou