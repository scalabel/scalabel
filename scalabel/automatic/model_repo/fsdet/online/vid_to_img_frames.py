from genericpath import exists
import os
import cv2
import multiprocessing as mp

# def split_video(file):
#     print("Processing file: {}".format(file))
#     out_dir = "/scratch/wcheng/bdd100k/val/frames/" + file.split(".")[0]
#     if not exists(out_dir):
#         os.mkdir(out_dir)

#     vidcap = cv2.VideoCapture(os.path.join(data_dir, file))
#     success,image = vidcap.read()
#     count = 0
#     while success:
#         out_file = os.path.join(out_dir, "{:06d}.png".format(count))
#         cv2.imwrite(out_file, image)      # save frame as JPEG file
#         success,image = vidcap.read()
#         # print('Read a new frame: ', count, end='\r')
#         count += 1

if __name__ == "__main__":
    data_dir = "/scratch/wcheng/bdd100k/val/videos/"
    # pool = mp.Pool()
    # pool.map(split_video, os.listdir(data_dir))

    for file in os.listdir(data_dir):
        print("Processing file: {}".format(file))
        out_dir = "/scratch/wcheng/bdd100k/val/frames/" + file.split(".")[0]
        if not exists(out_dir):
            os.mkdir(out_dir)

        vidcap = cv2.VideoCapture(os.path.join(data_dir, file))
        success,image = vidcap.read()
        count = 0
        while success:
            out_file = os.path.join(out_dir, "{:06d}.png".format(count))
            cv2.imwrite(out_file, image)     # save frame as JPEG file
            success,image = vidcap.read()
            print('Read a new frame: ', count, end='\r')
            count += 1
