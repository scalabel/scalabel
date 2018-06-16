import sys
import os
import re
from subprocess import Popen, PIPE, STDOUT
import json

FNULL = open(os.devnull, 'w')  # null file

video_directory_contents = os.listdir(sys.argv[1])

# make a directory
if not os.path.isdir(sys.argv[2]):
    os.mkdir(sys.argv[2])
frame_directory_contents = os.listdir(sys.argv[2])

# specify the frame rate if required
fps = int(sys.argv[3]) if len(sys.argv) > 3 else -1

for video_file in video_directory_contents:
    # if this isn't an .mp4 or .mov, continue
    if video_file[-4:] != '.mp4' and video_file[-4:] != '.mov':
        continue

    # if directory already exists, skip this file
    if video_file[:-4] in frame_directory_contents:
        print("Already processed: %s" % video_file)
        continue

    # get metadata using regex on the ffmpeg stderr
    cmd = "ffmpeg -i %s/%s" % (sys.argv[1], video_file)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
              close_fds=True)
    p.wait()
    pout = p.stdout.read().decode('utf-8')

    md = {}  # metadata
    md['bitrate'] = re.search("bitrate: ([0-9]* kb/s)", pout).group(1)
    md['fps'] = re.search(", ([0-9]*\\.?[0-9]*) fps,", pout).group(1) \
        if fps < 0 else fps
    md['tbr'] = re.search(", ([0-9]*\\.?[0-9]*) tbr,", pout).group(1) \
        if fps < 0 else fps
    md['tbn'] = re.search(", ([0-9]*) tbn,", pout).group(1)
    md['tbc'] = re.search(", ([0-9]*) tbc", pout).group(1)
    md['resolution'] = re.search(", ([0-9]{3,4}x[0-9]{3,4})", pout).group(1)

    output_directory = "%s/%s" % (sys.argv[2], video_file[:-4])
    os.mkdir(output_directory)

    # run ffmpeg to split into frames
    cmd = "ffmpeg -i %s/%s -r %s %s/f-%%07d.jpg" % (
        sys.argv[1], video_file, md['fps'], output_directory
    )
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    pout = p.stdout.read().decode('utf-8')
    p.wait()

    # get number of frames in final output
    nf = 0
    for file in os.listdir(output_directory):
        if file[-4:] == '.jpg':
            nf += 1
    md['numFrames'] = str(nf)

    # write metadata.json
    open("%s/metadata.json" % output_directory, 'w').write(json.dumps(md))

    print("Processed: %s" % video_file)

    # update the frame directory contents (shouldn't matter unless something
    #   like video.mp4 and video.mov in same folder)
    frame_directory_contents = os.listdir(sys.argv[2])
