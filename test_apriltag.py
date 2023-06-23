import apriltag
import cv2

img = cv2.imread('pics/'+'rgb_0.jpg', cv2.IMREAD_GRAYSCALE)
detector = apriltag.Detector()
result = detector.detect(img)
print(result)

corners = result[0].corners

for i in range(len(corners)):
    x, y = corners[i,:]
    img = cv2.circle(img, (int(x),int(y)), radius=5, color=(0, 0, 255), thickness=-1)

window_name = 'image'
  
# Using cv2.imshow() method
# Displaying the image
cv2.imshow(window_name, img)
  
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()