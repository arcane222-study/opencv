import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    * arcane222 (hong3883@naver.com)
    * Computer Vision assignment 2: Hough Transform

    * Dev. environment : Pycharm Community Edition 2019.3.3 -> 2022.06.11
                         opencv-contrib-python 
    * Interpreter: opencv-contrib-python 4.6.0.66
                   numpy 1.23.3
                   matplotlib 3.5.3
"""



""" 가로길이에 의한 이미지 크기 변경 (세로길이는 가로, 세로 비율에 맞게 조정)"""
def resize_by_width(src, width):
    dst = np.copy(src)
    h = dst.shape[0]
    w = dst.shape[1]
    dst = cv2.resize(dst, (width, int(width * h / w)), interpolation=cv2.INTER_CUBIC)
    return dst


""" src 를 gaussian filtering 을 해준 후 canny edge 를 검출하여 return"""
def make_canny_edge(src, threshold1, threshold2, k_size):
    dst = np.copy(src)
    dst = cv2.GaussianBlur(dst, k_size, 0)
    dst = cv2.Canny(dst, threshold1, threshold2)
    return dst


""" canny edge 영상을 가지고 hough transform 을 통해 직선을 검출하여 return"""
def make_hough_lines(edges, rho, theta, threshold):
    dst = np.copy(edges)
    lines = cv2.HoughLines(dst, rho, theta, threshold)
    return lines


""" 두점 (x1, y1), (x2, y2) 을 지나는 직선의 기울기(gradient) return"""
def get_line_gradient(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return 100000   # x 좌표가 같을 경우 y축에 수평인 직선이고 기울기는 무한대이기 때문에 매우 큰 숫자를 return 하여 맞춰줌
    return (y2 - y1) / (x2 - x1)


""" 두점 (x1, y1), (x2, y2) 을 지나는 직선의 y절편(y intercept) return """
def get_y_intercept(gradient, x1, y1):
    return y1 - gradient * x1


""" 각 두점 {(x1, y1), (x2, y2)} / {(x3, y3), (x4, y4)} 을 지나는 두 직선의 교점을 구함 """
def get_cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
    gradient1 = get_line_gradient(x1, y1, x2, y2)
    gradient2 = get_line_gradient(x3, y3, x4, y4)
    y_intercept1 = get_y_intercept(gradient1, x1, y1)
    y_intercept2 = get_y_intercept(gradient2, x3, y3)

    if gradient1 == gradient2:  # 두 직선이 평행한 경우 (두 직선이 완전히 일치하여 교점이 무수히 많은 경우도 포함)
        return None

    x = int((-1) * (y_intercept1 - y_intercept2) / (gradient1 - gradient2))
    y = int(gradient1 * x + y_intercept1)
    return x, y


""" lines (list type) 중 distance 값이 가장 작은 line 값을 반환 """
def get_min_dist_line(lines):
    size = len(lines)
    min_val = lines[0][0][0]
    idx = 0

    for i in range(1, size):
        dist = lines[i][0][0]
        if min_val > dist:
            min_val = dist
            idx = i

    return lines[idx]


""" lines (list type) 중 distance 값이 가장 큰 line 값을 반환 """
def get_max_dist_line(lines):
    size = len(lines)
    max_val = lines[0][0][0]
    idx = 0

    for i in range(1, size):
        dist = lines[i][0][0]
        if max_val < dist:
            max_val = dist
            idx = i

    return lines[idx]


"""  이미지의 둘레에 가장 유효한 직선 4개를 반환함 """
def get_significant_lines(lines):
    print("line cnt: " + str(len(lines)))
    row_lines = []
    column_lines = []
    significant_lines = []
    for line in lines:
        dist, theta = line[0]
        if 1 < theta < 2:
            row_lines.append(line)
        else:
            column_lines.append(line)

    print("row lines: " + str(len(row_lines)) + ", column lines: " + str(len(column_lines)))
    if len(row_lines) > 0:
        significant_lines.append(get_min_dist_line(row_lines))

    if len(row_lines) > 0:
        significant_lines.append(get_max_dist_line(row_lines))

    if len(column_lines) > 0:
        significant_lines.append(get_min_dist_line(column_lines))

    if len(column_lines) > 0:
        significant_lines.append(get_max_dist_line(column_lines))

    print("significant line cnt: " + str(len(significant_lines)) + '\n')
    return significant_lines


""" perspective transform 을 위한 점의 순서를 맞추기 위해 points 를 재정렬 하여 return 
    (왼쪽 위, 오른쪽 위, 왼쪽아래, 오른쪽 아래 순서) """
def reorder_points_to_fit_direction(points):
    point_list = []
    left_top = 0
    right_top = 0
    left_bottom = 0
    right_bottom = 0

    idx = 0
    val = points[0][0] + points[0][1]
    for i in range(1, len(points)):
        if points[i][0] + points[i][1] < val:
            val = points[i][0] + points[i][1]
            idx = i
    left_top = points[idx]
    points.pop(idx)

    idx = 0
    val = points[0][0] + points[0][1]
    for i in range(1, len(points)):
        if points[i][0] + points[i][1] > val:
            val = points[i][0] + points[i][1]
            idx = i
    right_bottom = points[idx]
    points.pop(idx)

    right_top = points[0] if points[0][0] > points[1][0] else points[1]
    left_bottom = points[0] if points[0][0] < points[1][0] else points[1]

    point_list.append(left_top)
    point_list.append(right_top)
    point_list.append(left_bottom)
    point_list.append(right_bottom)

    return point_list


""" 입력 이미지 src 와 src 의 모서리 4점 point_list 를 받아 화면에 가득 차도록 perspective transform 을 수행하여 return"""
def perspective_transformation(src, point_list):
    h, w = src.shape[:2]
    src_cornet_arr = np.float32(reorder_points_to_fit_direction(point_list))
    dst_corner_arr = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    transform = cv2.getPerspectiveTransform(src_cornet_arr, dst_corner_arr)
    dst = cv2.warpPerspective(src, transform, (w, h))
    return dst


""" 메인함수 (이미지를 입력받고 canny edge, hough lines 를 통해 4개의 모서리 점을 찾아 perspective transform 을 수행함 """
def main():
    # append input images to input_image_list
    size = 5
    input_img_list = []
    for i in range(size):
        input_img_list.append(cv2.imread("./img/test" + str(i) + ".png"))

    # append resize images to resize_img_list
    # (크기가 다른 이미지들로부터 비교적 균일한 값으로 직선 검출을 위해 크기를 통일함 가로 : 800, 세로 : 비율에 맞춤)
    resize_img_list = []
    for i in range(len(input_img_list)):
        resize_img_list.append(resize_by_width(input_img_list[i], 800))

    # append canny edge images to canny_edge_list & append hough lines to hough_lines_list
    canny_edge_list = []
    hough_lines_list = []
    for i in range(len(resize_img_list)):
        j = 0
        while True:
            canny = make_canny_edge(resize_img_list[i], 0, j * 10, (11, 11))
            hough_lines = make_hough_lines(canny, 1, np.pi / 180, 120)
            count = 0
            if hough_lines is not None:
                count = len(hough_lines)
                if i == 3 and 3 < count < 7:
                    canny_edge_list.append(canny)
                    hough_lines_list.append(hough_lines)
                    break
                if 3 < count < 5:
                    canny_edge_list.append(canny)
                    hough_lines_list.append(hough_lines)
                    break
            else:
                canny = make_canny_edge(resize_img_list[i], 0, (j - 1) * 10, (11, 11))
                hough_lines = make_hough_lines(canny, 1, np.pi / 180, 120)
                canny_edge_list.append(canny)
                hough_lines_list.append(hough_lines)
                break
            j = j + 1

        plt.imshow(canny_edge_list[i], cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.show()

    # 각 이미지별 lines 를 가지고 4개의 꼭짓점을 찾아내고 이것을 기반으로 perspective transformation 수행
    for i in range(len(hough_lines_list)):
        lines = hough_lines_list[i]
        significant_lines = []
        point_temp_list = []
        corner_point_list = []

        if lines is not None:
            # print(str(i + 1) + " : " + str(len(lines)))
            significant_lines = get_significant_lines(lines)    # dist 기준으로 min 가로, max 가로, min 세로, max 세로 선 4개 포함

        for line in significant_lines:
            dist, theta = line[0]

            cos_val = np.cos(theta)
            sin_val = np.sin(theta)
            x0 = cos_val * dist
            y0 = sin_val * dist
            x1 = int(x0 + 1000 * (-sin_val))
            y1 = int(y0 + 1000 * cos_val)
            x2 = int(x0 - 1000 * (-sin_val))
            y2 = int(y0 - 1000 * cos_val)
            point_temp_list.append((x1, y1, x2, y2))
            # cv2.line(resize_img_list[i], (x1, y1), (x2, y2), (0, 112, 255), 2)

        if len(point_temp_list) >= 3:
            for j in range(0, 2):
                for k in range(2, 4):
                    x1, y1, x2, y2 = point_temp_list[j]
                    x3, y3, x4, y4 = point_temp_list[k]
                    corner_point_list.append(get_cross_point(x1, y1, x2, y2, x3, y3, x4, y4))

            """for j in range(len(corner_point_list)):
                cv2.circle(resize_img_list[i], corner_point_list[j], 1, (0, 255, 0), 5)"""

            # cv2.imshow("line img " + str(i + 1), resize_img_list[i])

            # resize 한 이미지에 대하여 구한 4개의 모서리 점으로 perspective transform 을 수행함.
            dst = perspective_transformation(resize_img_list[i], corner_point_list)
            # perspective transform 한 이미지를 원본 이미지 크기로 다시 만듬
            dst = resize_by_width(dst, input_img_list[i].shape[1])
            # 결과이미지를 하나씩 출력
            plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
            plt.show()
            #cv2.imshow("Result Image " + str(i + 1), dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    main()