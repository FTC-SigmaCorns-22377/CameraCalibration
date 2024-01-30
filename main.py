
import numpy as np

import cv2

import glob

# Function to capture images for calibration
def capture_images():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Capture Images', cv2.WINDOW_NORMAL)

    print("Press 'c' to capture image, 'q' to quit")

    num_images = 10
    image_count = 0

    while image_count < num_images:
        ret, frame = cap.read()

        cv2.imshow('Capture Images', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            image_count += 1
            print(f"Image {image_count} captured.")
            cv2.imwrite(f'calibration_image_{image_count}.png', frame)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to perform camera calibration
def calibrate_camera():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')

    images = [cv2.imread(f'calibration_image_{i}.png') for i in range(1, 11)]

    for img in images:
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

if __name__ == "__main__":
    capture_images()
    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = calibrate_camera()

    if ret:
        print("Calibration successful!")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(distortion_coefficients)

        # Use the camera matrix and distortion coefficients
        img = cv2.imread('left12.jpg')
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))
        dst = cv2.undistort(img, camera_matrix, distortion_coefficients, None, newcameramtx)
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('calibresult.png', dst)
    else:
        print("Calibration failed.")
