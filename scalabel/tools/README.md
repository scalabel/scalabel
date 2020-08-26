# Tools for processing Scalabel inputs and outputs

## prepare_data

This code converts videos or images to a data folder and an image list that can be directly used for creating Scalabel projects. Assume all the images are in `.jpg` format.

You can download our testing video at https://scalabel-public.s3-us-west-2.amazonaws.com/demo/dancing.mp4. The video was from [Youtube](https://www.youtube.com/watch?v=-ZTsgbrdoI8) with Creative Commons Attribution license.

Convert video to a folder of images and generate the image list:

```bash
python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100 \
    -i dancing.mp4 -o ./dancing --fps 3 \
    --url-root http://localhost:8686/items/dancing1k
```

The above command also starts extracting the frames from the 5th second and
stops at 100th frame. The extraction fps is 3 and it is assumed that the
images are served from the `items` dir.

If you want to upload the images to the s3 and generate the corresponding image
list, you can do

```bash
python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100 \
    -i ~/Downloads/dancing.mp4  --fps 3 -o dancing-s3 \
    --s3 scalabel-public/demo/dancing
```

We also support multiple inputs, as well as image folder. The command below
will read the images from the folder `dancing-s3` and copy them to the output
folder before breaking down the video. The script will think we will create a frame list of two videos. The images in the generated list will be assigned to one of two video names.

```bash
python3 -m scalabel.tools.prepare_data --start-time 5 --max-frames 100  \
    --fps 3 -i ./dancing-s3 dancing.mp4  -o test
```

## edit_labels

`edit_labels` provides utilities to edit image and label list.

Add url prefix to the name field in the frames and assign it to the url field

```bash
python3 -m scalabel.tools.edit_labels --add-url http://localhost:8686/items -i \
    input.json -o output.json
```
