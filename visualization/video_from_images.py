import cv2
import os
import time

def create_video(image_folder, image_prefix, video_name, frame_rate, repeat_to):
    images = [img for img in os.listdir(image_folder) if img.startswith(image_prefix) and img.endswith(".png")]
    images.sort()

    print("Total images for the video: {}".format(len(images)))
    if len(images) < repeat_to:
        # Repeat the last image to make the video longer
        for i in range(repeat_to - len(images)):
            images.append(images[-1])
    # print the number of total images
    print("Total images with repeated last frame: {}".format(len(images)))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

    print("Creating video...")
    time_start = time.time()
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    time_end = time.time()
    print("Done!")
    print("Time elapsed: {}s".format(time_end - time_start))
    video.release()

if __name__ == "__main__":
    # image_folder = '/Users/shiqing/Desktop/Visual-Servoing-DOB/results_collision_dob/exp_004_wo_cbf'
    image_folder = '/Users/shiqing/Desktop/Visual-Servoing-DOB/results_collision_dob_sphere/exp_002_w_spheres'
    # image_prefix = 'screenshot'  # Images should start with this prefix
    image_prefix = 'rgb'  # Images should start with this prefix
    video_name = image_folder + '/' + image_prefix + '_video.mp4'
    frame_rate = 10
    repeat_to = 130

    create_video(image_folder, image_prefix, video_name, frame_rate, repeat_to)