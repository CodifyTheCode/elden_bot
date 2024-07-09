import cv2
import numpy as np
import sys
import os

def index_to_key(inds):
    vals = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
    keys = [vals[i] for i, val in enumerate(inds) if val == 1]
    return keys

def load_batch(folder, init, end):
    try:
        ins = np.load(f"{folder}/{init}_{end}.npy")
        images = np.load(f"{folder}/img{init}_{end}.npy")
        rewards = np.load(f"{folder}/r{init}_{end}.npy")
        return ins, images, rewards
    except FileNotFoundError:
        return None, None, None

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_images1"
    batch_size = 100
    init = 0
    end = batch_size
    i = 0

    while True:
        if i == init:
            ins, images, rewards = load_batch(folder, init, end)
            if ins is None:
                print("End of Playback")
                break

        keys = index_to_key(ins[i - init])
        print(keys, rewards[i - init])
        img = images[i - init]
        cv2.imshow("test", img)
        
        i += 1
        if i == end:
            init += batch_size
            end += batch_size

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
