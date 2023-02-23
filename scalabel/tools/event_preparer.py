# Script to load the event data and generate representations of the events

from utils.eventreader import EventReader
import numpy as np
import os
from pathlib import Path
from utils.eventslicer import EventSlicer
import h5py
from PIL import Image, ImageOps

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

def padding(img, expected_h, expected_w):
    delta_width = expected_w - img.size[0]
    delta_height = expected_h - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding,(255,255,255))


class EventPreparer:
    def __init__(self, filepath: Path, event_time_delta_us, event_cam_h,event_cam_w) -> None:
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self.time_delta_us = event_time_delta_us
        self.event_slicer = EventSlicer(self.h5f)
        self.t_start_us = self.event_slicer.get_start_time_us()
        self.t_end_us = self.event_slicer.get_final_time_us()
        self.event_cam_h = event_cam_h
        self.event_cam_w = event_cam_w
        
    def get_start_time_us(self):
        return self.t_start_us

    def get_final_time_us(self):
        return self.t_end_us

    def create_representations(self,timestamps, path, pad_size_h=None,pad_size_w=None):
        for i, t in enumerate(timestamps):
            events = self.event_slicer.get_events(t, t+self.time_delta_us)
            p = events['p']
            x = events['x']
            y = events['y']
            t = events['t']
            img = render(x, y, p, self.event_cam_h, self.event_cam_w) 
            if pad_size_h is not None and pad_size_w is not None:
                img = np.array(padding(Image.fromarray(img), pad_size_h, pad_size_w))
            h, w = img.shape[:2]
            img_alpha = np.dstack((img,np.zeros((h,w),dtype=np.uint8)+255))
            mWhite = (img_alpha[:,:,0:3] == [255,255,255]).all(2)
            img_alpha[mWhite] = (0,0,0,0)
            im = Image.fromarray(img_alpha)
            
            im.save(str(path/("event_rep_" + str(i).zfill(6)+ ".png")))
            print("Event Representation generated: " + str(i)+ " out of "+ str(len(timestamps)))