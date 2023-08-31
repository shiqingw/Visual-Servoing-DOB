import cv2
import numpy as np
  
img1 = cv2.imread('results_dob/exp_002/rgb_0000.png')
img2 = cv2.imread('results_dob/exp_002/scaling_functions_0000.png')

ind = np.where(img2 >1)
img2[ind] = img1[ind]

# img2 = cv2.resize(img2, img1.shape[1::-1])

alpha = 0.5

dst = cv2.addWeighted(img1, alpha , img2, 1-alpha, 0)

cv2.imwrite('alpha_mask_.png', dst)

  