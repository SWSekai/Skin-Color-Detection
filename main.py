import cv2
import numpy as np
import os

def is_skin(r, g, b, h, s, y, cr, cb):
    """
    判斷一個像素是否為膚色。
    """
    rgb_condition = (
        r > 95 and g > 40 and b > 20 and r > g and r > b and abs(r - g) > 15
    )
    hsv_condition = s > 15
    ycbcr_condition = (
        cr > 135
        and cb > 85
        and y > 80
        and cr <= (1.5862 * cb) + 20
        and cr >= (0.3448 * cb) + 76.2069
        and cr >= (-4.5652 * cb) + 234.5652
        and cr <= (-1.15 * cb) + 301.75
        and cr <= (-2.2857 * cb) + 432.85
    )

    return rgb_condition and hsv_condition and ycbcr_condition


def skin_detect(image_path, ground_truth_path):
    """
    偵測圖片中的膚色區域並計算 IOU。
    """
    
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)


    height, width, channels = image.shape

    skin_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x] # BGR

            # 5. Convert RGB to HSV
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv[0], hsv[1], hsv[2]

            # 6. Convert RGB to YCbCr
            ycbcr = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2YCrCb)[0][0]
            yc, cr, cb = ycbcr[0], ycbcr[1], ycbcr[2]  # YCrCb

            if is_skin(r, g, b, h, s, yc, cr, cb):
                # 8. Do not mask the pixel
                skin_mask[y, x] = image[y, x] # 保留原始像素顏色
            else:
                # 9. Mask the pixel with black color
                skin_mask[y, x] = [0, 0, 0] # 將非膚色像素設為黑色

    # IOU 計算
    intersection = np.logical_and(ground_truth, cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY))
    union = np.logical_or(ground_truth, cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY))
    iou = np.sum(intersection) / np.sum(union)

    return skin_mask, iou


if __name__ == "__main__":
    image_folder = "./Photo"
    ground_truth_folder = "./GroundTruth"

    image_files = sorted(os.listdir(image_folder))
    ground_truth_files = sorted(os.listdir(ground_truth_folder))

    total_iou = 0
    for image_file, ground_truth_file in zip(image_files, ground_truth_files):
        image_path = os.path.join(image_folder, image_file)
        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_file)

        skin_masked_image, iou = skin_detect(image_path, ground_truth_path)
        total_iou += iou

        print(f"Image: {image_file}, IOU: {iou}")

        # cv2.imshow(f"Skin Detection: {image_file}", skin_masked_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    average_iou = total_iou / len(image_files)
    print(f"\nAverage IOU: {average_iou}")