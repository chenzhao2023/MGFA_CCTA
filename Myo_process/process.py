import cv2
import numpy as np
import nibabel as nib

def get_dilated_contour(binary_image):

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return binary_image

    contour_image = np.zeros_like(binary_image)
    contour_image = np.ascontiguousarray(contour_image, dtype=np.uint8)


    for contour in contours:

        leftmost_idx = contour[:, :, 0].argmin()
        bottommost_idx = contour[:, :, 1].argmax()
        leftmost = tuple(contour[leftmost_idx][0])
        bottommost = tuple(contour[bottommost_idx][0])

        new_contour = []
        for i in range(len(contour)):
            if not (min(leftmost_idx, bottommost_idx) <= i <= max(leftmost_idx, bottommost_idx)):
                new_contour.append(contour[i])

        new_contour = np.array(new_contour, dtype=np.int32)

        cv2.drawContours(contour_image, [new_contour], -1, 255, thickness=1)

    kernel_size = 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 扩张轮廓
    dilated_contour = cv2.dilate(contour_image, kernel, iterations=1)
    # cv2.imshow("Dilated Contour (Outward Expansion)", dilated_contour)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dilated_contour



if __name__ == '__main__':

    patient = r'12019951'
    numbers = 120
    # 读取二值图像和原始图像
    binary_image_path = rf'F:\UltraLight-VM-UNet-main\UltraLight-VM-UNet-main\test1020-f5\{patient}_image_slice_{numbers}.png_pred.png'
    original_image_path = rf'F:\UltraLight-VM-UNet-main\UltraLight-VM-UNet-main\test1020-f5\{patient}_image_slice_{numbers}.png'
    vessel_label = nib.load(rf'D:\imgcas\{patient}\label.nii.gz').get_fdata().transpose(2,1,0)
    print(vessel_label.shape)
    vessel_label = np.flip(vessel_label, axis=1)
    vessel_label = vessel_label[numbers,:,:]
    # vessel_label = np.transpose(vessel_label,(1,0),)

    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    if binary_image is None or original_image is None:
        raise ValueError("Failed to load images. Please check the file paths.")

    dilated_contour = get_dilated_contour(binary_image)
    original_image_resized = original_image

    custom_color2 = [255, 215, 0]
    ct_image_colored = cv2.cvtColor(original_image_resized, cv2.COLOR_GRAY2BGR)
    vessel_color = np.zeros_like(ct_image_colored)
    vessel_color[vessel_label > 0] = custom_color2

    ct_image = cv2.addWeighted(ct_image_colored, 0.7, vessel_color, 1, 0)  # 0.5 表示半透明程度


    # 显示结果
    # cv2.imshow("Modified Contour (No Arc + Line)", contour_image)
    # cv2.imshow("Modified Contour (No Arc + Line", contour_ori)
    # cv2.imshow("Modified Contour (No Arc + Lin", binary_image)
    cv2.imshow("Dilated Contour (Outward Expansion)", dilated_contour)
    cv2.imshow("Original Image (Resized)", ct_image)
    # cv2.imshow("Original Image (Resized)", vessel_label)
    # 将灰度图像转换为3通道图像，以便能够进行半透明处理
    ct_image_colored = cv2.cvtColor(original_image_resized, cv2.COLOR_GRAY2BGR)


    # 定义自定义颜色 (例如红色：BGR = [0, 0, 255])
    custom_color2 = [255, 215, 0]  # BGR 格式
    custom_color = [0, 0, 255]

    # 创建彩色 mask，并将 dilated_contour 中白色部分填充为自定义颜色
    dilated_contour = dilated_contour - vessel_label
    mask_colored = np.zeros_like(ct_image_colored)  # 创建一个与原图相同大小的空彩色图像
    mask_colored[dilated_contour == 255] = custom_color  # 将 dilated_contour 中的白色部分应用为自定义颜色

    vessel_color = np.zeros_like(ct_image_colored)
    vessel_color[vessel_label > 0] = custom_color2
    # 将彩色 mask 应用于原始图像，生成半透明效果
    ct_image_colored = cv2.addWeighted(ct_image_colored, 1, mask_colored, 0.2, 0)  # 0.5 表示半透明程度
    ct_image_colored = cv2.addWeighted(ct_image_colored, 0.7, vessel_color, 1, 0)  # 0.5 表示半透明程度

    # # 定义渐变颜色（例如从蓝色到绿色）
    # color_start = np.array([255, 0, 0])  # 起始颜色 (红色 - BGR)
    # color_end = np.array([0, 255, 255])  # 结束颜色 (黄色 - BGR)
    #
    # # 创建与原图大小一致的空彩色图像
    # mask_colored = np.zeros_like(ct_image_colored, dtype=np.uint8)
    #
    # # 获取图像尺寸
    # height, width = dilated_contour.shape
    #
    # # 生成渐变颜色
    # for y in range(height):
    #     # 计算渐变比例
    #     alpha = y / height
    #     # 根据比例插值计算当前行的颜色
    #     color = (1 - alpha) * color_start + alpha * color_end
    #     # 将当前行填充到彩色 mask 中
    #     mask_colored[y, :] = color
    #
    # # 将 dilated_contour 的白色部分保留为渐变颜色
    # mask_colored[dilated_contour == 0] = 0  # 黑色部分保留为 0
    #
    # # 将彩色 mask 应用于原始图像，生成半透明效果
    # masked_image = cv2.addWeighted(ct_image_colored, 1, mask_colored, 0.5, 0)  # 0.5 表示透明度

    # 显示结果
    cv2.imshow("mask_image", ct_image_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # # 创建形态学操作的卷积核（越大填补越多）
    # kernel = np.ones((15, 15), np.uint8)
    # # dilated_image = cv2.dilate(image, kernel, iterations=4)
    # # # 执行形态学闭运算 (先膨胀后腐蚀)
    # closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #
    #
    # # 使用 OpenCV 展示原图和填补后的图像
    # cv2.imshow('Original Image', image)  # 显示原图
    # cv2.imshow('Closed Image', closed_image)  # 显示闭运算后的图像



