import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
    * arcane222 (hong3883@naver.com)
    * Computer Vision assignment 3: Image Stitching

    * Dev. environment : Pycharm Community Edition 2019.3.3 -> 2022.06.11
                         opencv-contrib-python 
    * Interpreter: opencv-contrib-python 4.6.0.66
                   numpy 1.23.3
                   matplotlib 3.5.3
                   
    * Issue : OpenCV 3.4.2.16 이상에서는 SIRF, SURF를 지원하지 않음.
             (최신버전으로 구동 시 아래와 같은 에러메세지를 확인할 수 있음)
             ((-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration; 
             Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'cv::xfeatures2d::SURF::create')
             
             1. OpenCV를 3.4.2.16 이하 버전 사용
             or
             2. cv2.xfeatures2d.SURF_create() 대신 cv2.SIFT_create() 사용
"""


""" src 에 대하여 grayscale image 를 return """
def make_gray_image(src):
    dst = np.copy(src)
    return cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


""" src 에 대하여 key point list 와 descriptor list 를 return """
def create_descriptor(src, threshold):
    # SURF = cv2.xfeatures2d.SURF_create()
    # SURF.setHessianThreshold(threshold)

    SURF = cv2.SIFT_create(edgeThreshold=threshold)

    key_point, descriptor = SURF.detectAndCompute(src, None)
    return key_point, descriptor


""" 두개의 descriptor list 를 입력받아 matching list 를 반환"""
def matching(descriptor1, descriptor2):
    brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    match_list = brute_force_matcher.match(descriptor1, descriptor2)
    return sorted(match_list, key=lambda x: x.distance)


""" 두 입력 이미지의 유사한 key point 를 매칭시켜주고 이를 통해 Homography 값을 구하여 우측 이미지의 perspective transform 수행 """
def perspective_transformation(src_left, src_right, match_list, key_point_left, key_point_right):
    key_point_left2 = []
    key_point_right2 = []

    for match in match_list:
        queryIdx = match.queryIdx
        trainIdx = match.trainIdx
        key_point_left2.append(np.float32(key_point_left[queryIdx].pt))
        key_point_right2.append(np.float32(key_point_right[trainIdx].pt))

    key_point_left2 = np.reshape(key_point_left2, (-1, 1, 2))
    key_point_right2 = np.reshape(key_point_right2, (-1, 1, 2))

    h, status = cv2.findHomography(key_point_right2, key_point_left2, cv2.RANSAC, 4.0)
    dst = cv2.warpPerspective(src_right, h, (src_left.shape[1] + src_right.shape[1], src_right.shape[0]))
    return dst


""" src1, src2(perspective transform 된 오른쪽 이미지) 두개를 이어붙임 """
def image_stitching(src1, src2):
    dst = np.copy(src2)
    dst[:src1.shape[0], :src1.shape[1]] = src1
    return dst


""" 합친 이미지의 우측에 생기는 검은색 영역을 제거함 (합쳐진 이미지를 전체 이미지 크기로 perspective transformation) """
def remove_black_space(src):
    h, w = src.shape[:2]
    left_top = 0
    right_top = 0
    left_bottom = 0
    right_bottom = 0
    src_gray = make_gray_image(src)
    src_gray = cv2.medianBlur(src_gray, 3)

    ret, thresh = cv2.threshold(src_gray, 127, 255, 0)
    # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = - 1
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    approx = cv2.approxPolyDP(best_cnt, 0.01 * cv2.arcLength(best_cnt, True), True)
    left_top = [0, 0]
    x_list = []
    y_list = []
    for ap in approx:
        y_list.append(ap[0, 1])
        x_list.append(ap[0, 0])
        if ap[0, 0] == 0 and ap[0, 1] != 0:
            left_bottom = [ap[0, 0], ap[0, 1]]
        if ap[0, 1] == 0 and ap[0, 0] != 0:
            right_top = [ap[0, 0], ap[0, 1]]

    right_bottom = [max(x_list), max(y_list)]

    src_corner_arr = np.float32([left_top, right_top, left_bottom, right_bottom])
    dst_corner_arr = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    transform = cv2.getPerspectiveTransform(src_corner_arr, dst_corner_arr)
    dst = cv2.warpPerspective(src, transform, (src.shape[1], src.shape[0]))

    return dst


def main():
    # 입력 이미지와 grayscale 이미지
    input_left = cv2.imread('./img/left.jpg')
    input_right = cv2.imread('./img/right.jpg')
    input_left_gray = make_gray_image(input_left)
    input_right_gray = make_gray_image(input_right)

    # grayscale 이미지로부터 얻어낸 key point 들의 list 와 descriptor 들의 list
    key_point_left, descriptor_left = create_descriptor(input_left_gray, 100)
    key_point_right, descriptor_right = create_descriptor(input_right_gray, 100)
    
    # 두 이미지의 매칭되는 점들에 대한 정보를 담은 matching list 
    # (queryIdx: 왼쪽 이미지 특징점 의 idx, trainIdx: 왼쪽 이미지의 특징점에 매칭되는 오른쪽 이미지의 특징점의 idx, distance: 두 점간 거리)
    match_list = matching(descriptor_left, descriptor_right)
    
    # 오른쪽 이미지에 대하여 perspective transform 수행 후 두 이미지 합체
    dst_perspective = perspective_transformation(input_left, input_right, match_list, key_point_left, key_point_right)
    dst_stitching = image_stitching(input_left, dst_perspective)

    # 두 이미지 간 특징점을 연결한 결과 이미지
    dst_match_result = np.zeros((input_left.shape[0], input_left.shape[1] * 2), input_left.dtype)
    dst_match_result = cv2.drawMatches(input_left, key_point_left, input_right, key_point_right, match_list[0:30], dst_match_result, flags=2)

    plt.imshow(cv2.cvtColor(dst_match_result, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(dst_perspective, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(dst_stitching, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(remove_black_space(dst_stitching), cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
