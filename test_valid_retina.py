import os
import random
import cv2
import numpy as np

def is_valid_retina(image_path):
    img = cv2.imread(image_path)
    if img is None: return False
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    b, g, r = avg_color
    if r < g + 5 or r < b + 5:
        return False
    return True

classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
dataset_dir = "dataset"

passed = 0
failed = 0
for c in classes:
    class_dir = os.path.join(dataset_dir, c)
    if not os.path.exists(class_dir): continue
    images = os.listdir(class_dir)
    for img_name in random.sample(images, min(10, len(images))):
        img_path = os.path.join(class_dir, img_name)
        if is_valid_retina(img_path):
            passed += 1
        else:
            failed += 1

print(f"Passed: {passed}, Failed: {failed}")
