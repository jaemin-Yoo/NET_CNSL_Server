import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


point_list = []
count = 0

#GPS list
gps_list = []
gps_list.append((35.832909,128.754458)) # Left Up
gps_list.append((35.832842,128.754476)) # Right Up
gps_list.append((35.832850,128.754155)) # Left Down
gps_list.append((35.832776,128.754171)) # Right Down

#픽셀 간 위도 경도 비
gps_width = (math.sqrt((gps_list[0][0]-gps_list[1][0])**2+(gps_list[0][1]-gps_list[1][1])**2)+math.sqrt((gps_list[2][0]-gps_list[3][0])**2+(gps_list[2][1]-gps_list[3][1])**2))/2
gps_length = (math.sqrt((gps_list[0][0]-gps_list[2][0])**2+(gps_list[0][1]-gps_list[2][1])**2)+math.sqrt((gps_list[1][0]-gps_list[3][0])**2+(gps_list[1][1]-gps_list[3][1])**2))/2
pixel_width = 800
pixel_length = 800
width_rate = gps_width / pixel_width
length_rate = gps_length / pixel_length
#사람 발끝 좌표 * rate = gps 좌표

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original


    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)



cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

# 원본 이미지
temp = cv2.imread('test.jpg')
img_original = cv2.resize(temp, (800, 800))


while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break


# 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

print(pts1)
print(pts2)

M = cv2.getPerspectiveTransform(pts1,pts2)

img_result = cv2.warpPerspective(img_original, M, (width,height))
#res_img = cv2.resize(img_result,(800,800))

plt.imshow(img_result)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
