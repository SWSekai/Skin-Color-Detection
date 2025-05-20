import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from method import skin_detect

def createWindow(image = None, title = None): 
    """
    創建一個圖形視窗
    """
    plt.figure(figsize=(10, 6))
    
    if image is not None:
        plt.imshow(image, cmap='gray')
        
    if title is not None:
        plt.title(title)
    
    # 去除圖形標框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.xticks([]), plt.yticks([]) # 隱藏坐標軸刻度

def main():
    sum_iou = 0
    sum_iou = 0
    
    for i in range(1, 7):
        image_path = f"./Photo/pic{i}.jpg"
        ground_truth_path = f"./GroundTruth/pic{i}.png"
        origin_result_path = f"./Result/pic{i}.jpg"
        
        skin_masked, iou = skin_detect(image_path, ground_truth_path)

        print(f"picture{i} image\'s IOU: {iou}")

        
        createWindow(title=f"picture{i} image\'s IOU: {iou.round(4)}")
        
        plt.subplot(1, 3, 1), plt.title("Image"), plt.imshow(cv2.imread(image_path)), plt.axis("off"), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 3, 2), plt.title("Ground truth"), plt.imshow(cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE), cmap="gray"), plt.axis("off"), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 3, 3), plt.title("Result"), plt.imshow(skin_masked, cmap="gray"), plt.axis("off"), plt.xticks([]), plt.yticks([])
        
        output_dir = "./Result_Compared"
        output_path = os.path.join(output_dir, f"pic{i}.png")
        plt.savefig(output_path, bbox_inches='tight')
        
        plt.show()
        
        cv2.imwrite(f"./Result/pic{i}.jpg", skin_masked)
        
        sum_iou += iou
    
    average_iou = sum_iou/6
    
    print(f"\nAverage IOU is: {average_iou}")

if __name__ == "__main__":

    main()