import numpy as np
import cv2
import random

# bilinear interpolation
def my_bilinear(img, x, y):
    '''
    :param img: 값을 찾을 img
    :param x: interpolation 할 x좌표
    :param y: interpolation 할 y좌표
    :return: img[x,y]에서의 value (bilinear interpolation으로 구해진)
    '''
    floorX, floorY = int(x), int(y)

    t, s = x - floorX, y - floorY

    zz = (1 - t) * (1 - s)
    zo = t * (1 - s)
    oz = (1 - t) * s
    oo = t * s

    interVal = img[floorY, floorX, :] * zz + img[floorY, floorX + 1, :] * zo + \
               img[floorY + 1, floorX, :] * oz + img[floorY + 1, floorX + 1, :] * oo

    return interVal

def backward(img1, M):
    '''
    :param img1: 변환시킬 이미지
    :param M: 변환 matrix
    :return: 변환된 이미지
    '''
    h, w, c = img1.shape
    result = np.zeros((h, w, c))

    for row in range(h):
        for col in range(w):
            xy_prime = np.array([[col, row, 1]]).T
            xy = (np.linalg.inv(M)).dot(xy_prime)

            x_ = xy[0, 0]
            y_ = xy[1, 0]

            if x_ < 0 or y_ < 0 or (x_ + 1) >= w or (y_ + 1) >= h:
                continue

            result[row, col, :] = my_bilinear(img1, x_, y_)

    return result



def my_ls(matches, kp1, kp2):
    '''
    :param matches: keypoint matching 정보
    :param kp1: keypoint 정보.
    :param kp2: keypoint 정보2.
    :return: X : 위의 정보를 바탕으로 Least square 방식으로 구해진 Affine
                변환 matrix의 요소 [a, b, c, d, e, f].T
    '''
    A = []
    B = []
    for idx, match in enumerate(matches):
        trainInd = match.trainIdx
        queryInd = match.queryIdx
        x, y = kp1[queryInd].pt
        x_prime, y_prime = kp2[trainInd].pt

        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append([x_prime])
        B.append([y_prime])

    A = np.array(A)
    B = np.array(B)

    try:
        ATA = np.dot(A.T, A)
        ATB = np.dot(A.T, B)
        X = np.dot(np.linalg.inv(ATA), ATB)
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X


def get_matching_keypoints(img1, img2, keypoint_num):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: 추출한 keypoint의 수
    :return: img1의 특징점인 kp1, img2의 특징점인 kp2, 두 특징점의 매칭 결과
    '''
    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.DIST_L2)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    """
    matches: List[cv2.DMatch]
    cv2.DMatch의 배열로 구성

    matches[i]는 distance, imgIdx, queryIdx, trainIdx로 구성됨
    trainIdx: 매칭된 img1에 해당하는 index
    queryIdx: 매칭된 img2에 해당하는 index

    kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    """
    return kp1, kp2, matches

def feature_matching(img1, img2, keypoint_num=None):
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    X = my_ls(matches, kp1, kp2)

    M = np.array([[X[0][0], X[1][0], X[2][0]],
                  [X[3][0], X[4][0], X[5][0]],
                  [0, 0, 1]])

    result = backward(img1, M)
    return result.astype(np.uint8)

def feature_matching_RANSAC_Gaussian(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=10):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: sift에서 추출할 keypoint의 수
    :param iter_num: RANSAC 반복횟수
    :param threshold_distance: RANSAC에서 inlier을 정할때의 거리 값
    :return: RANSAC을 이용하여 변환 된 결과
    '''
    img21=cv2.resize(img2,None,fx=0.5,fy=0.5)
    kp1, kp2, matches = get_matching_keypoints(img1, img21, keypoint_num)

    matches_shuffle = matches.copy()

    inliers = [] #랜덤하게 고른 n개의 point로 구한 inlier개수 결과를 저장
    M_list = [] #랜덤하게 고른 n개의 point로 만든 affine matrix를 저장
    for i in range(iter_num):
        print('\rcalculate RANSAC ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        #######################################################################
        # ToDo 
        # RANSAC을 이용하여 최적의 affine matrix를 찾고 변환하기
        # 1. 랜덤하게 3개의 matches point를 뽑아냄
        # 2. 1에서 뽑은 matches를 이용하여 affine matrix M을 구함
        # 3. 2에서 구한 M을 모든 matches point와 연산하여 inlier의 개수를 파악
        # 4. iter_num 반복하여 가장 많은 inlier를 가지는 M을 최종 affine matrix로 채택
        ########################################################################
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]

        try :
            X = my_ls(three_points, kp1, kp2)
            M = np.array([[X[0][0], X[1][0], X[2][0]],
                          [X[3][0], X[4][0], X[5][0]],
                          [0, 0, 1]])
        except:
            inliers.append(0)
            M_list.append([])
            continue

        inliers.append(0)
        for j in range(0,len(matches)):
            idx1=matches[j].queryIdx

            (x1, y1) = kp1[idx1].pt[0],kp1[idx1].pt[1]
            arr=np.array([[x1],[y1],[1]])

            x=np.dot(M,arr)
            vec1 = np.array([[int(x[0,0])],[int(x[1,0])]])
            vec2 = np.array([[int(kp2[matches[j].trainIdx].pt[0])], [int(kp2[matches[j].trainIdx].pt[1])]])
            dist = L2_distance(vec1,vec2)
            if(dist<threshold_distance):
               inliers[i]+=1
        M_list.append(M)

    max_in = np.argmax(inliers)
    best_M = M_list[max_in]
    result = backward(img1, best_M)
    ########################################################################
    result=GaussianFiltering(result)
    new_img=np.zeros(img2.shape)
    h1,w1,c1=new_img.shape
    h2,w2,c2=result.shape
    rate=h1/w2

    for i in range(h1):
        for j in range(w1):
            x, y = (j / rate), (i / rate)
            floorX, floorY = int(x), int(y)

            t, s = x - floorX, y - floorY

            zz = (1 - t) * (1 - s)
            zo = t * (1 - s)
            oz = (1 - t) * s
            oo = t * s
            up = min(int(np.round(floorY)), int(h2) - 1)
            down = min(int(np.round(floorY+1)), h2 - 1)
            left = min(int(np.round(floorX)), w2 - 1)
            right = min(int(np.round(floorX+1)), w2 - 1)
            interVal = result[up, left, :] * zz + result[up, right, :] * zo + \
                       result[down, left, :] * oz + result[down, right, :] * oo

            new_img[i, j, :] = interVal
    ########################################################################

    return new_img.astype(np.uint8)

def L2_distance(vector1, vector2):

    return np.sqrt(np.sum((vector1-vector2)**2))

def my_padding(src, filter):
    h, w, cnt = src.shape

    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2, cnt))

    for i in range(cnt) :
        padding_img[h_pad:h + h_pad, w_pad:w + w_pad, i] = src[:, :, i]
       # repetition padding
        # up
        padding_img[:h_pad, w_pad:w_pad + w,i] = src[0, :,i]
        # down
        padding_img[h_pad + h:, w_pad:w_pad + w,i] = src[h - 1, :,i]
        # left
        padding_img[:, :w_pad,i] = padding_img[:, w_pad:w_pad + 1,i]
        # right
        padding_img[:, w_pad + w:,i] = padding_img[:, w_pad + w - 1:w_pad + w,i]

    return padding_img

def my_filtering(src, filter):
    (h, w,t) = src.shape
    (f_h, f_w) = filter.shape

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)
    dst = np.zeros((h, w, t))
    for i in range(t):
        for row in range(h):
            for col in range(w):
                dst[row, col,i] = np.sum(pad_img[row:row + f_h, col:col + f_w,i] * filter)

    return dst

def my_get_Gaussian_filter(fshape, sigma=0.5):
    (f_h, f_w) = fshape
    y, x = np.mgrid[-(f_h // 2):(f_h // 2) + 1, -(f_w // 2):(f_w // 2) + 1]
    #2차 gaussian mask 생성
    filter_gaus = 1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    filter_gaus /= np.sum(filter_gaus)

    return filter_gaus

def GaussianFiltering(src, fshape = (5,5), sigma=0.5):
    gaus = my_get_Gaussian_filter(fshape, sigma)
    dst = my_filtering(src, gaus)
    return dst


def main():
    src = cv2.imread('../Lena.png')
    src=cv2.resize(src,None,fx=0.5,fy=0.5)
    src2 = cv2.imread('../LenaFaceShear.png')
    result_RANSAC = feature_matching_RANSAC_Gaussian(src, src2)
    cv2.imshow('input', src)
    cv2.imshow('gaussian result  ', result_RANSAC)
    cv2.imshow('goal', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
