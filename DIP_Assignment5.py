import cv2
import numpy as np

def skin_detect(image_path, ground_truth_path):
    """
    偵測圖片中的膚色區域並計算 IOU。
    """

    # 讀取圖片
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path)

    # 顏色空間轉換
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # 3. 膚色範圍設定
    lower_rgb = np.array([0, 40, 80], dtype=np.uint8)
    upper_rgb = np.array([255, 255, 255], dtype=np.uint8)

    lower_hsv = np.array([0, 40, 30], dtype=np.uint8)
    upper_hsv = np.array([43, 255, 255], dtype=np.uint8)

    lower_ycbcr = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycbcr = np.array([255, 180, 135], dtype=np.uint8)

    # 4. 膚色檢測
    mask_rgb = cv2.inRange(image, lower_rgb, upper_rgb)
    mask_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    mask_ycbcr = cv2.inRange(image_ycbcr, lower_ycbcr, upper_ycbcr)

    # 綜合三個顏色空間的結果
    skin_mask = cv2.bitwise_and(mask_rgb, cv2.bitwise_and(mask_hsv, mask_ycbcr))

    # 5. IOU 計算
    intersection = np.logical_and(ground_truth, skin_mask)
    union = np.logical_or(ground_truth, skin_mask)
    iou = np.sum(intersection) / np.sum(union)

    return iou

if __name__ == '__main__':
    sum_iou = 0
    
    for i in range (1, 7):
        image_path = f"./Photo/pic{i}.jpg"
        ground_truth_path = f"./GroundTruth/pic{i}.png"
        
        iou = skin_detect(image_path, ground_truth_path)
        print(f'picture{i} image\'s IOU: {iou}')
        
        sum_iou += iou
    
    average_iou = sum_iou/6
    
    print(f"\nAverage IOU is: {average_iou}")