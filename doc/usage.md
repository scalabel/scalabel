# User Guide

## Data Preparation 
Our tool supports both video and image annotations. We provide a script at
`scripts/prepare_data.py` to help you prepare data.  

- For the video data, we will first split the video into frames at the specified
frame rate (default is 5fps) and then generate a yaml file containing the paths 
to the frames. 
```
python3 scripts/prepare_data.py -i <path_to_video> -t <output_directory> -f <fps>
--s3 <s3_bucket_name/folder>
```
Flag `--s3` is optional. If you want to upload the output frames and the 
frame list to your Amazon S3 bucket, you can specify the key to the bucket. 
We assume the bucket is public readable so that the generated links
can be directly accessed by the tool. 
Please refer to the [doc](http://boto3.readthedocs.io/en/latest/guide/s3-example-creating-buckets.html)
to configure your S3 bucket. 

- For the image data,  you need to specify the path to the folder that contains
the images. If `--s3` is specified, the images will be uploaded to the S3 bucket
together with the image list yaml file that contains the urls to the images.
```
python3 scripts/prepare_data.py -i <path_to_image_folder>  --s3 <s3_bucket_name/folder>
```


Once obtaining the `image_list.yml`, we can proceed to create a new annotation project. 


## Serving Data
If you wish to serve your own data on the same domain as this server, you may
 do so. This means the item_list.yml you provide at project creation can 
 contain URLs to this server. Any files or directories you place in "app/src/" 
 will be uploaded to the server with the prefix path removed. For example, 
 you could place a directory "images" in "app/src/". If you are just testing
 the server locally, you can then specify in your item list URLs such as 
 "localhost:8686/images/image_1.jpg".

## Launch Server 
As provided in the [README](../README.md), you can either set up the tool with
the provided docker image or build the tool by yourself. Once you have the tool setup, 
you can open `http://localhost:8686` to create your annotation project.


## Create Project 

The interface of creating a new project is show as follows. We use the video 
bounding box annotation as an example.

![alt text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/create_project_video.png)

In this case, you need to upload `image_list.yml` which contains either
paths to the video frames or public readable urls to the frames in the `Item List`.
For `Categories` and `Attributes`, you can upload the yaml files that contains the
labels and attributes descriptions respectively. Please refer to `examples/` folder
for more details. 

At this point, you can click `Go to Dashboard` to monitor the project, download
results and try the annotation tool.  The interface is shown as follows. 

![alt text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/video_dashboard.png)

If you click `task link`,  you can go to the annotation interface and 
start your work. 

![alt text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/annotation_interface_a.png)



## 2D Bounding Box 
For tasks like object detection, we need to draw bounding boxes for the
objects of interest in the image. The interface of drawing 2D bounding boxes
is shown as follows. 

![alt text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/image_bbox_a.png)

On the left side, we show all the label categories specified in `categories.yml`
and also provide `Occluded` and `Truncated` options to indicate whether or not
the object is occluded or truncated. For traffic lights, we also have the options
to indicate the color of the light on the left side.  

When drawing the bounding boxes, it's better to zoom in the image by
clicking the `+` button on the top right if the object
of interest in too small.  The zoom-in image is shown as follows. You can 
use your mouse to move the canvas to annotate different regions. If you want 
to remove one bounding box, simply click the bounding box to highlight it and
then click `Remove` button on the bottom left. 

![alt text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/image_bbox_zoom_in_a.png)

Once finish drawing all the bounding boxes, you can save the results by 
clicking the `Save` button on the top right.  Then the bounding boxes results
will be saved to your data folder specified in the `configure.yml`. 

A short demo of drawing 2D bounding boxes is shown below. Please check out the 
video with higher resolution [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/2d_bbox_caption.mp4).

<p align="center">
  <img width="720" height="480" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/2d_bbox_caption_ll.gif">
</p>


## Instance Segmentation
The tool also supports the instance segmentation annotation. The interface
is similar as the interface of 2D bounding boxes where on the left side, we
provide label categories and other attributes of the labels. 

To draw a mask of the instance, we draw multiple anchor points along the boundary
of the instance and end at the initial point to complete the drawing. 
![alt_text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/inst_seg_done.png)

A short demo of drawing instance segmentation is shown below.  Please check out the video with higher 
resolution [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/inst_seg_caption.mp4).

<p align="center">
  <img width="720" height="504" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/inst_seg_caption_ll.gif">
</p>

Besides, you can enter the **quick draw** mode by clicking the `Quickdraw` on the bottom left 
or pressing `s` which allows the two instances to share the common border. 
Check out the video demo with higher resolution [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/quickdraw_demo.mp4).

<p align="center">
  <img width="720" height="504" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/quickdraw_demo_h.gif">
</p>


Our tool also allows you to **bezier the curve** to improve the annotation quality. 
Checkout the video demo [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/bezier_demo.mp4).

<p align="center">
  <img width="720" height="504" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/bezier_demo_h.gif">
</p>

### Lane Marking

Lane marking is used in a similar way as the other image annotations, but you will need to select a different 
label type when you are creating your project. The operations are the same as the instance segmentation. 
In this case, you won’t see a whole chuck of color filled in your selected area, since lane labeling is just 
outlining different sections of a street.

Here is a short demo of lane marking below. Checkout the video with higher resolution [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/lane_caption.mp4).
<p align="center">
  <img width="720" height="480" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/lane_caption_h.gif">
</p>

### Drivable Area

Similarly as other instance segmentation task, you can annotate the drivable area in the image. 
Details about the drivable area can be found in this [blog](http://bair.berkeley.edu/blog/2018/05/30/bdd/).

An example annotation is shown below.
![alt_text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/drivable_area.png)



## Video Tracking 
We also support video object tracking. Firstly, we draw a bounding box around
the object of interest at the starting frame. 

![alt_text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/video_tracking_start.png)

Then, continue playing the video until the ending frame. Move the bounding box
to the ending position of the object and adjust the bounding box size. 

![alt_text](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/screenshots/video_tracking_end.png)

At this point, if you replay the video, the bounding box will track the object
and adjust its size across frames. 

Here is a short demo to track a person in the video. Please check out the video with higher
resolution [here](https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/video_tracking_caption.mp4).

<p align="center">
  <img width="720" height="504" src="https://s3-us-west-2.amazonaws.com/scalabel-public/demo/videos/video_tracking_caption_ll.gif">
</p>







