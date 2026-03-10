import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=10):
    """在图像上添加高斯噪声"""
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_motion_blur(image, kernel_size=15, angle=0):
    """对图像添加运动模糊"""
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = 1
    # 旋转核
    M = cv2.getRotationMatrix2D((kernel_size/2 -0.5, kernel_size/2 -0.5), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel /= kernel.sum()
    # 应用卷积
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def degrade_image(image_path, mode='gaussian', **kwargs):
    """
    对图片添加退化
    :param image_path: 输入图片路径
    :param mode: 'gaussian' 或 'motion'
    :param kwargs: 高斯噪声: mean, sigma; 运动模糊: kernel_size, angle
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片，请检查路径")

    if mode == 'gaussian':
        mean = kwargs.get('mean', 0)
        sigma = kwargs.get('sigma', 10)
        return add_gaussian_noise(img, mean, sigma)
    elif mode == 'motion':
        kernel_size = kwargs.get('kernel_size', 15)
        angle = kwargs.get('angle', 0)
        return add_motion_blur(img, kernel_size, angle)
    else:
        raise ValueError("mode 参数只能是 'gaussian' 或 'motion'")

# ===== 示例使用 =====
if __name__ == "__main__":
    input_path = "/apdcephfs_fsgm/share_304156246/xmudongwang/codebase/zq/YOLOv8-Crack-Detection/test_datasets/clean/valid/images/00792_jpg.rf.4006a8b7ae7585ab0522270c8a285e24.jpg"  # 输入图片路径
    output_path = "00792_jpg.rf.4006a8b7ae7585ab0522270c8a285e24_motion.jpg"

    # 选择退化类型
    degraded = degrade_image(input_path, mode='gaussian', mean=0, sigma=100)
    # degraded = degrade_image(input_path, mode='motion', kernel_size=20, angle=30)

    cv2.imwrite(output_path, degraded)
    print(f"退化图像已保存到 {output_path}")