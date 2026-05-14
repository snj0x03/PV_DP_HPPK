# extact frame function
import yaml
import cv2, os, shutil

import pandas as pd

from PIL import Image

with open("../config.yml", "r") as file:
    data = yaml.safe_load(file)




def extract_frames(video_path, save_dir, frame_rate, max_frames, hash_threshold, seen_hashes, part_number):

    if seen_hashes is None:
        seen_hashes = []

    cap = cv2.VideoCapture(video_path)
    fps = 30
    interval = max(int(fps * frame_rate * 2), 1)
    

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        if max_frames is not None and saved >= max_frames:
            break
        if count % interval == 0:
            try:
                if hash_threshold is not None:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    h = imagehash.phash(pil_img)
                    is_duplicate = any(abs(h - seen)) <= hash_threshold for seen in seen_hashes)
                else:
                    is_duplicate = False
                if not is_duplicate:
                    fname = f"{part_number}-{saved}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), frame)
                    if hash_threshold is not None:
                        seen_hashes.append(h)
                    saved += 1
            except Exception as e:
                print(e)
        count += 1
    cap.release()
    return saved
                            
                           
