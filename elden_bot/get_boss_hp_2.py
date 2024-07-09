import cv2
import os
import numpy as np
import time

def load_images_from_folder(folder):
    images = [cv2.imread(os.path.join(folder, filename)) 
              for filename in os.listdir(folder) if cv2.imread(os.path.join(folder, filename)) is not None]
    return images

def process_image(img, left, down, width, height):
    img2 = img[left:left + width, down:down + height]
    _, threshb = cv2.threshold(img2[:, :, 0], 190, 255, cv2.THRESH_BINARY)
    _, threshg = cv2.threshold(img2[:, :, 1], 190, 255, cv2.THRESH_BINARY)
    _, threshr = cv2.threshold(img2[:, :, 2], 190, 255, cv2.THRESH_BINARY)
    summed = threshb + threshg + threshr
    value = np.max(np.argmax(summed, axis=1))
    return value, img2, threshb, threshg, threshr

if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), "test_images")
    imgs = load_images_from_folder(folder_path)

    left = 908
    down = 495
    width = 8
    height = 1067

    for img in imgs:
        start_time = time.time()

        value, img2, threshb, threshg, threshr = process_image(img, left, down, width, height)

        end_time = time.time()
        print(value)
        print(f"Processing time: {end_time - start_time} seconds")

        cv2.imshow("original", img)
        cv2.imshow("roi", img2)
        cv2.imshow("redChannel", threshr)
        cv2.imshow("blueChannel", threshb)
        cv2.imshow("greenChannel", threshg)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
