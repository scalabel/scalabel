import os
import re
from subprocess import Popen, PIPE, STDOUT
import json

FNULL = open(os.devnull, 'w')  # null file

for vidfile in os.listdir("data/videos"):
    # if this isn't an mp4, continue
    if vidfile[-4:] != '.mp4' and vidfile[-4:] != '.mov':
        continue

    # if directory already exists, skip this file
    if(vidfile[:-4] in os.listdir("data/frames")):
        print("Already processed: %s" % vidfile)
        continue

    # get metadata using regex on the ffmpeg stderr
    cmd = "ffmpeg -i data/videos/%s" % vidfile
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
              stderr=STDOUT, close_fds=True)
    p.wait()
    pout = p.stdout.read().decode('utf-8')

    md = {}  # metadata
    md['bitrate'] = re.search("bitrate: ([0-9]* kb/s)", pout).group(1)
    md['fps'] = re.search(", ([0-9]*\\.?[0-9]*) fps,", pout).group(1)
    md['tbr'] = re.search(", ([0-9]*\\.?[0-9]*) tbr,", pout).group(1)
    md['tbn'] = re.search(", ([0-9]*) tbn,", pout).group(1)
    md['tbc'] = re.search(", ([0-9]*) tbc", pout).group(1)
    md['resolution'] = re.search(", ([0-9]{3,4}x[0-9]{3,4})", pout).group(1)

    # make a directory
    frame_dir = "data/frames/%s" % vidfile[:-4]
    os.mkdir(frame_dir)

    # run ffmpeg to split into frames
    cmd = "ffmpeg -i data/videos/%s -r %s %s/f-%%07d.jpg" % (
        vidfile, md['fps'], frame_dir)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    pout = p.stdout.read().decode('utf-8')
    p.wait()

    # get number of frames in final output
    # md['numFrames'] = re.search("frame=[ ]+([0-9]+)[ ]+fps", pout).group(1)
    nf = 0
    for file in os.listdir(frame_dir):
        if file[-4:] == '.jpg':
            nf += 1
    md['numFrames'] = str(nf)

    # write metadata.json
    open("%s/metadata.json" % frame_dir, 'w').write(json.dumps(md))

    print("Processed: %s" % vidfile)
