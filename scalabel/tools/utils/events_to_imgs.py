import argparse
from pathlib import Path

import numpy as np
import skvideo.io
from tqdm import tqdm

from eventreader import EventReader


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Visualize Events')
#     parser.add_argument('event_file', type=str, help='Path to events.h5 file')
#     parser.add_argument('output_file', help='Path to write video file')
#     parser.add_argument('--delta_time_ms', '-dt_ms', type=float, default=50.0, help='Time window (in milliseconds) to summarize events for visualization')
#     args = parser.parse_args()

#     event_filepath = Path(args.event_file)
#     video_filepath = Path(args.output_file)
#     dt = args.delta_time_ms

#     height = 480
#     width = 640

#     assert video_filepath.parent.is_dir(), "Directory {} does not exist".format(str(video_filepath.parent))

#     writer = skvideo.io.FFmpegWriter(video_filepath)
#     for events in tqdm(EventReader(event_filepath, dt)):
#         p = events['p']
#         x = events['x']
#         y = events['y']
#         t = events['t']
#         img = render(x, y, p, height, width)
#         writer.writeFrame(img)
#     writer.close()