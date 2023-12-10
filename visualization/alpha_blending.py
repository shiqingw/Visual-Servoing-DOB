import cv2
import numpy as np
import os

results_path = "results_collision_dob/"
exp_num = "exp_013_keep"
image_folder = results_path + exp_num

images = [img for img in os.listdir(image_folder) if img.startswith("scaling") and img.endswith(".png")]
images.sort()

alpha = 0.5

for i in range(len(images)):
    img_scaling = cv2.imread(os.path.join(image_folder, images[i]))
    num = images[i].split("_")[-1].split(".")[0]
    img_rgb = cv2.imread(os.path.join(image_folder, "rgb_" + num + ".png"))
    ind = np.where(img_scaling >1)
    img_scaling[ind] = img_rgb[ind]
    dst = cv2.addWeighted(img_rgb, alpha , img_scaling, 1-alpha, 0)
    cv2.imwrite(os.path.join(image_folder, "blending_" + num + ".png"), dst)


  