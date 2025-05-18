import cv2
import numpy as np

def skin_detect_optimized(image_path, ground_truth_path=None):
    
    image = cv2.imread(image_path)
    if ground_truth_path:
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # 影像前處理 (平滑)
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 顏色空間轉換
    image_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)
    image_ycbcr = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2YCrCb)

    # 膚色檢測 (論文中的條件 + 條件組合)
    skin_mask = np.zeros_like(image, dtype=np.uint8)[:, :, 0]  # 單通道遮罩

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j, 2], image[i, j, 1], image[i, j, 0]
            h, s, _ = image_hsv[i, j]
            y, cr, cb = image_ycbcr[i, j]

            # 論文中的膚色條件
            rgb_condition = r > 97 and g > 40 and b > 20 and r > g and r > b and abs(int(r) - int(g)) > 15
            hsv_condition = 0.0 <= h <= 50.0 and 0.23 <= s <= 0.68
            ycbcr_condition = cr > 135 and cb > 85 and y > 80 and cr <= (1.5862 * cb) + 20 and cr >= (0.3448 * cb) + 76.2069 and cr >= (-4.5652 * cb) + 234.5652 and cr <= (-1.15 * cb) + 301.75 and cr <= (-2.2857 * cb) + 432.85

            # 條件組合
            if rgb_condition or (hsv_condition and ycbcr_condition):
                skin_mask[i, j] = 255

    # 後處理 (形態學操作)
    kernel = np.ones((3, 3), np.uint8)
    skin_mask_opened = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask_closed = cv2.morphologyEx(skin_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 邊緣檢測
    edges = cv2.Canny(image_blurred, 100, 200)
    skin_mask = cv2.bitwise_or(skin_mask_closed, edges)  # 將邊緣加入遮罩

    # 效能評估
    iou = None
    if ground_truth_path:
        iou = calculate_iou(ground_truth, skin_mask_closed)

    return skin_mask_closed, iou


def calculate_iou(mask1, mask2):
    """計算兩個二值化遮罩的 IOU"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou