import cv2
import numpy as np


img = cv2.imread("sample.png")

def threshold_otsu(gray, min_value=0, max_value=255):
    # calc histgrams
    hist = [np.sum(gray == i) for i in range(256)]
    s_max = (0,-10)

    for th in range(256):
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        # calc average number of pixel for each class
        mu1 = 0
        mu2 = 0
        if n1 != 0:
            mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 != 0:
            mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        s = n1 * n2 * (mu1 - mu2) ** 2
        if s > s_max[1]:
            s_max = (th, s)
    
    t = s_max[0]
    gray[gray==t] = max_value
    return gray


def erode(img):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(img,kernel,iterations = 1)

def dilate(img):
    kernel = np.ones((2,2),np.uint8)
    return cv2.dilate(img,kernel,iterations = 1)

def opening(img):
    kernel = np.ones((2,2),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img):
    kernel = np.ones((2,2),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def main():
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th_img = threshold_otsu(gray)
    cv2.imwrite("th.png", th_img)
    res = cv2.bitwise_not(gray)

    res = erode(res)
    res = dilate(res)
    res = opening(res)
    res = dilate(res)
    res = threshold_otsu(res)

    # res = closing(res)
    # res = closing(res)

    fixed_img = cv2.bitwise_not(res)
    cv2.imwrite("er.png", fixed_img)

if __name__ == "__main__":
    main()    
