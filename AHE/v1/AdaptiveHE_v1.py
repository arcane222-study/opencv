import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    * arcane222 (hong3883@naver.com)
    * Image Processing assignment 2: Adaptive Histogram Equalization
    
    * Dev. environment : Pycharm Community Edition 2019.3.3 -> 2022.06.11
                         opencv-contrib-python 
    * Interpreter: opencv-contrib-python 4.6.0.66
                   numpy 1.23.3
                   matplotlib 3.5.3
"""
# 201621093 이홍준
# Assignment2 Adaptive HE

def get_neighborhood_hist(_src, cy, cx):  # 3x3
    height = _src.shape[0]
    width = _src.shape[1]

    left_x = cx - 1
    right_x = cx + 1
    top_y = cy - 1
    bottom_y = cy + 1

    if left_x < 0 or left_x > width - 1:
        left_x = cx
    if right_x < 0 or right_x > width - 1:
        right_x = cx
    if top_y < 0 or top_y > height - 1:
        top_y = cy
    if bottom_y < 0 or bottom_y > height - 1:
        bottom_y = cy

    mask = np.zeros(_src.shape, _src.dtype)
    mask[top_y:bottom_y + 1, left_x:right_x + 1] = 255

    local_hist = cv2.calcHist([_src], [0], mask, [256], [0, 256])
    local_hist[_src[cy, cx], 0] -= 1  # Remove Center Pixel

    return local_hist


def get_hist_mean(neighborhood_hist, arg, size):
    val_sum = 0
    for i in range(neighborhood_hist.shape[0]):
        if neighborhood_hist[i, 0] > 0:
            val_sum += np.power(i, arg) * neighborhood_hist[i, 0]
    return val_sum / size


def get_hist_std(neighborhood_hist, size):
    mean1 = get_hist_mean(neighborhood_hist, 2, size)
    mean2 = get_hist_mean(neighborhood_hist, 1, size)
    return np.sqrt(mean1 - mean2 * mean2)


def hist_equalization(src, src_hist):
    intensity_max = 256
    height = src.shape[0]
    width = src.shape[1]

    if height * width == 0:
        return

    cdf = ((intensity_max - 1) / (height * width)) * np.cumsum(src_hist)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i, j] = np.uint8(np.round(cdf[src[i, j]]))


def hist_redistribution(img_input, clip_limit):
    intensity_max = 256
    height = img_input.shape[0]
    width = img_input.shape[1]
    hist = cv2.calcHist([img_input], [0], None, [256], [0, 256])
    if clip_limit <= 0 or clip_limit >= 256:
        return hist
    clip_limit_val = np.floor(height * width * clip_limit / 256)
    excess_tmp = 0

    for i in range(intensity_max):
        if hist[i, 0] > clip_limit_val:
            excess_tmp += hist[i, 0] - clip_limit_val
            hist[i, 0] = clip_limit_val
    for i in range(intensity_max):
        if hist[i, 0] + np.floor(excess_tmp / intensity_max) <= clip_limit_val:
            hist[i, 0] += np.floor(excess_tmp / intensity_max)
    return hist


def local_hist_equalization(src, size, clip_limit, idx):
    height = int(np.ceil(src.shape[0] / size))
    width = int(np.ceil(src.shape[1] / size))
    dst = np.zeros(src.shape, src.dtype)
    progress = 0

    # Copy Input Image to Output Image
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = src[i, j]

    for i in range(size):
        for j in range(size):
            local = dst[i * height:(i + 1) * height, j * width:(j + 1) * width]
            hist_local = hist_redistribution(local, clip_limit)
            hist_equalization(dst[i * height:(i + 1) * height, j * width:(j + 1) * width], hist_local)
            progress = progress + 1
            print("LHE1 - " + str(int(100 * progress / (size * size))) + "%, No. " + str(idx))
    return dst


def local_hist_equalization2(_src, idx):
    height = _src.shape[0]
    width = _src.shape[1]
    dst = np.zeros(_src.shape, _src.dtype)
    progress = 0
    hist_global = cv2.calcHist([_src], [0], None, [256], [0, 256])
    mean_global = get_hist_mean(hist_global, 1, height * width)
    std_global = get_hist_std(hist_global, height * width)

    for i in range(height):
        for j in range(width):
            hist_local = get_neighborhood_hist(_src, i, j)
            mean_local = get_hist_mean(hist_local, 1, 8)
            std_local = get_hist_std(hist_local, 8)

            condition1 = mean_local <= 0.4 * mean_global
            condition2 = 0.02 * std_global <= std_local <= 0.4 * std_global
            if condition1 and condition2:
                dst[i, j] = 4 * _src[i, j]
            else:
                dst[i, j] = _src[i, j]

        progress = progress + width
        print("LHE 2 - " + str(int(100 * progress / (width * height))) + "%, No. " + str(idx))
    return dst


def main():
    size = 4

    src = []
    for i in range(size):
        src.append(cv2.imread("./img/test" + str(i) + ".png", cv2.IMREAD_GRAYSCALE))

    dst = []
    dst.append(local_hist_equalization(src[0], 1, 64, 0))
    dst.append(local_hist_equalization(src[1], 1, 64, 1))
    dst.append(local_hist_equalization(src[2], 103, 255, 2))
    dst.append(local_hist_equalization2(src[3], 3))

    for i in range(len(dst)):
        plt.imshow(dst[i], cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
