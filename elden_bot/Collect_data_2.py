import time
import cv2
import mss
import numpy as np
import sys
import os
import keyboard
import win32api as wapi

# Constants
KEYS = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
KEY_LIST = ["\b"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\")
SAVE_BATCH_SIZE = 100
WIDTH = 244
NEW_HEIGHT = 138
LEFT = 908
DOWN = 495
HEIGHT = 1067
SCREEN_MONITOR = {"top": 50, "left": 0, "width": 2048, "height": 1152}

# Create index dictionary
indexes = {key: idx for idx, key in enumerate(KEYS)}

def key_check():
    return [key for key in KEY_LIST if wapi.GetAsyncKeyState(ord(key))] + \
           ([" "] if wapi.GetAsyncKeyState(0) else [])

def end(start_time, count):
    print("Done")
    duration = time.time() - start_time
    print(f"Average FPS: {count / duration}")

def keys_to_array(keys):
    x = np.zeros(11, dtype=np.uint8)
    for key in keys:
        if key in indexes:
            x[indexes[key]] = 1
    return x

if __name__ == "__main__":
    folderString = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    path = os.path.join(os.getcwd(), folderString)

    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        temp = 1
        while os.path.isdir(path):
            path = os.path.join(os.getcwd(), f"{folderString}{temp}")
            temp += 1
        folderString += str(temp - 1)
        os.mkdir(path)

    init_count = 0
    key_ins = np.zeros((SAVE_BATCH_SIZE, 11), dtype=np.uint8)
    images = np.zeros((SAVE_BATCH_SIZE, NEW_HEIGHT, WIDTH), dtype=np.uint8)
    rewards = np.zeros((SAVE_BATCH_SIZE, 1), dtype=np.float16)
    max_health = None

    time.sleep(5)
    start_time = time.time()

    with mss.mss() as sct:
        try:
            count = 0
            while True:
                img = np.array(sct.grab(SCREEN_MONITOR))
                img2 = img[LEFT : LEFT + WIDTH, DOWN : DOWN + HEIGHT]

                # Threshold and find health value
                threshb = cv2.threshold(img2[:, :, 0], 190, 255, cv2.THRESH_BINARY)[1]
                threshg = cv2.threshold(img2[:, :, 1], 190, 255, cv2.THRESH_BINARY)[1]
                threshr = cv2.threshold(img2[:, :, 2], 190, 255, cv2.THRESH_BINARY)[1]
                summed = threshb + threshg + threshr
                value = np.max(np.argmax(summed, axis=1))
                if count == 0:
                    max_health = value
                value = -value / max_health

                # Process and store images and key inputs
                img = cv2.resize(img, (WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                images[count % SAVE_BATCH_SIZE] = img

                keys = key_check()
                key_ins[count % SAVE_BATCH_SIZE] = keys_to_array(keys)
                rewards[count % SAVE_BATCH_SIZE] = value

                count += 1
                if count % SAVE_BATCH_SIZE == 0 and count > 0:
                    np.save(f"{folderString}/{init_count}_{count}", key_ins)
                    np.save(f"{folderString}/img{init_count}_{count}", images)
                    np.save(f"{folderString}/r{init_count}_{count}", rewards)
                    init_count = count

        except KeyboardInterrupt:
            end(start_time, count)
