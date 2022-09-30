import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    * arcane222 (hong3883@naver.com)
    * Computer Vision assignment 1: Multi-band Blending

    * Dev. environment : Pycharm Community Edition 2019.3.3 -> 2022.06.11
                         opencv-contrib-python 
    * Interpreter: opencv-contrib-python 4.6.0.66
                   numpy 1.23.3
                   matplotlib 3.5.3
"""


# 가우시안 피라미드를 얻는 함수 (리스트형태로 반환)
# 입력이미지, 레벨값, 커널의 사이즈를 매개변수로 입력받음
def get_gaussian_pyramid(src, level, k_size):
    pyramid_list = []
    dst = src.copy()
    pyramid_list.append(dst)

    for i in range(level):
        # 각 단계의 이미지를 절반만큼 다운스케일 하고 가우시안 필터링을 적용
        resize_h = int(dst.shape[0] / 2)
        resize_w = int(dst.shape[1] / 2)
        gaussian_img = cv2.GaussianBlur(dst, (k_size, k_size), 0)
        resize_img = cv2.resize(gaussian_img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

        # 필터된 이미지를 가우시안 피라미드 리스트에 추가하고 dst값을 현재 피라미드 단계의 이미지로 갱신
        pyramid_list.append(resize_img)
        dst = resize_img.copy()

    return pyramid_list


# 라플라시안 피라미드를 얻는 함수 (리스트형태로 반환)
# 입력이미지, 레벨값, 커널의 사이즈를 매개변수로 입력받음
def get_laplacian_pyramid(src, level, k_size):
    pyramid_list = []
    gaussian_list = get_gaussian_pyramid(src, level, k_size)
    pyramid_list.append(gaussian_list[level])

    for i in range(level):
        idx = level - 1 - i
        dst1 = gaussian_list[idx + 1].copy()
        dst2 = gaussian_list[idx].copy()

        # 업스케일한 이미지와 같은레벨의 이미지 크기를 비교해서 이미지 크기를 맞춰줌
        resize_h = dst1.shape[0] * 2 if dst1.shape[0] * 2 == dst2.shape[0] else dst2.shape[0]
        resize_w = dst1.shape[1] * 2 if dst1.shape[1] * 2 == dst2.shape[1] else dst2.shape[1]

        # 이미지를 가로 세로 두배의 길이로 업스케일함
        reconstruction = cv2.resize(dst1, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

        # 가우시안 된 이미지에서 업스케일한 이미지를 빼 라플라시안 이미지를 얻고 리스트에 넣어 피라미드를 만듬
        laplacian_img = cv2.subtract(dst2, reconstruction)
        pyramid_list.append(laplacian_img)

    return pyramid_list


# 라플라시안 피라미드와 가우시안 피라미드를 각 적용한 입력 이미지 두개와 마스크에 대하여 블렌딩 하는 함수
# 매개변수: 입력이미지 1, 입력이미지 2, 마스크이미지, 레벨값, 커널의 사이즈
def blending(src1, src2, mask, level, k_size):

    # src1, src2의 라플라시안 피라미드 리스트 & mask의 가우시안 피라미드 리스트, 블렌딩한 각 레벨의 이미지의 리스트
    img1_laplacian_list = get_laplacian_pyramid(src1, level, k_size)
    img2_laplacian_list = get_laplacian_pyramid(src2, level, k_size)
    mask_list = get_gaussian_pyramid(mask, level, k_size)
    blending_list = []

    # 각 피라미드의 값을 a b m으로 가져온 후 식 계산을 위해 float64로 형변환후 식을 계산하여 blending 이미지 리스트를 얻음
    for i in range(level + 1):
        idx = level - i
        a = np.float64(img1_laplacian_list[i])
        b = np.float64(img2_laplacian_list[i])
        m = np.float64(mask_list[idx])
        result = np.zeros(a.shape, a.dtype)

        for j in range(result.shape[0]):
            for k in range(result.shape[1]):
                for l in range(result.shape[2]):
                    result[j, k, l] = ((255 - m[j, k, l]) * a[j, k, l] + m[j, k, l] * b[j, k, l]) / 255

        # 0 ~ 255로 정규화 후 데이터를 uint8로 형변환한 후 리스트에 추가
        cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(result)
        blending_list.append(result)
        
    # 각 블랜딩된 이미지를 가져와 두배로 업스케일한 이미지와 더해 결과 이미지를 획득함
    blending_img = blending_list[0].copy()
    for i in range(level):
        dst1 = blending_list[i + 1].copy()
        
        # 업스케일한 이미지와 같은레벨의 이미지 크기를 비교해서 이미지 크기를 맞춰줌
        resize_h = blending_img.shape[0] * 2 if blending_img.shape[0] * 2 == dst1.shape[0] else dst1.shape[0]
        resize_w = blending_img.shape[1] * 2 if blending_img.shape[1] * 2 == dst1.shape[1] else dst1.shape[1]
        
        # 이미지를 가로 세로 두배의 길이로 업스케일함
        reconstruction = cv2.resize(blending_img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        blending_img = cv2.add(reconstruction, dst1)

    return blending_img


# 이미지 3개를 읽어와 blending_img 변수에 결과값을 담고 imshow를 통해 출력
def main():
    img_apple = cv2.imread("./img/burt_apple.png", cv2.IMREAD_COLOR)
    img_orange = cv2.imread("./img/burt_orange.png", cv2.IMREAD_COLOR)
    img_mask = cv2.imread("./img/burt_mask.png", cv2.IMREAD_COLOR)

    level = 6
    kernel_size = 3
    blending_img = blending(img_orange, img_apple, img_mask, level, kernel_size)

    plt.imshow(cv2.cvtColor(blending_img, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
