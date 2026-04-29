import torch
from PIL import Image
import numpy as np
from inference import analyze_eye_image

# 1. Generate Black Image
black_img = Image.new('RGB', (500, 500), color = 'black')
black_img.save('black_test.jpg')

# 2. Generate White Image
white_img = Image.new('RGB', (500, 500), color = 'white')
white_img.save('white_test.jpg')

# 3. Generate Random Noise
noise = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
noise_img = Image.fromarray(noise)
noise_img.save('noise_test.jpg')

print("Testing Black Image:")
print(analyze_eye_image('black_test.jpg'))

print("\nTesting White Image:")
print(analyze_eye_image('white_test.jpg'))

print("\nTesting Noise Image:")
print(analyze_eye_image('noise_test.jpg'))
