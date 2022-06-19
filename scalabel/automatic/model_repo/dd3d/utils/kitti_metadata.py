from detectron2.data import DatasetCatalog, MetadataCatalog
import seaborn as sns
from scalabel.automatic.model_repo.dd3d.utils.visualization import float_to_uint8_color
from collections import OrderedDict

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
COLORMAP = OrderedDict(
    {
        "Car": COLORS[2],  # green
        "Pedestrian": COLORS[1],  # orange
        "Cyclist": COLORS[0],  # blue
        "Van": COLORS[6],  # pink
        "Truck": COLORS[5],  # brown
        "Person_sitting": COLORS[4],  #  purple
        "Tram": COLORS[3],  # red
        "Misc": COLORS[7],  # gray
    }
)

metadata = MetadataCatalog.get("kitti_3d")
metadata.thing_classes = VALID_CLASS_NAMES
metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]
metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
