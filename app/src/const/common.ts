/**
 * Constant type definition
 */

export enum LabelTypeName {
  EMPTY = "empty",
  TAG = "tag",
  BOX_2D = "box2d",
  POLYGON_2D = "polygon2d",
  POLYLINE_2D = "polyline2d",
  CUSTOM_2D = "custom2d",
  BOX_3D = "box3d",
  PLANE_3D = "plane3d"
}

export enum ItemTypeName {
  IMAGE = "image",
  VIDEO = "video",
  POINT_CLOUD = "pointcloud",
  POINT_CLOUD_TRACKING = "pointcloudtracking",
  FUSION = "fusion"
}

export enum HandlerUrl {
  INVALID = "NO_VALID_HANDLER",
  LABEL = "label"
}

export enum BundleFile {
  V2 = "image_v2.js",
  V1 = "image.js"
}

export enum ShapeTypeName {
  UNKNOWN = "unknown",
  RECT = "rect",
  CUBE = "cube",
  GRID = "grid",
  POINT_2D = "point2d",
  PATH_POINT_2D = "path_point2d",
  NODE_2D = "node2d",
  POLYGON_2D = "polygon2d"
}

export enum DataType {
  IMAGE = "image",
  IMAGE_3D = "image_3d",
  POINT_CLOUD = "pointcloud"
}

export enum ViewerConfigTypeName {
  IMAGE = "image",
  POINT_CLOUD = "pointcloud",
  IMAGE_3D = "image_3d",
  HOMOGRAPHY = "homography",
  UNKNOWN = "unknown",
  RADAR = "radar"

}

export enum PathPointType {
  LINE = "line",
  CURVE = "bezier" // Cubic Bezier curve path points
}

export enum Cursor {
  CROSSHAIR = "crosshair",
  DEFAULT = "default",
  EW_RESIZE = "ew-resize",
  NESW_RESIZE = "nesw-resize",
  NWSE_RESIZE = "nwse-resize",
  NS_RESIZE = "ns-resize",
  MOVE = "move",
  POINTER = "pointer"
}

export enum Key {
  ESCAPE = "Escape",
  CONTROL = "Control",
  META = "Meta",
  ALT = "Alt",
  SHIFT = "Shift",
  BACKSPACE = "Backspace",
  SPACE = " ",
  ENTER = "Enter",
  PERIOD = ".",
  SLASH = "/",
  EQUAL = "=",
  PLUS = "+",
  MINUS = "-",
  DOWN = "Down",
  UP = "Up",
  LEFT = "Left",
  RIGHT = "Right",
  ARROW_DOWN = "ArrowDown",
  ARROW_UP = "ArrowUp",
  ARROW_LEFT = "ArrowLeft",
  ARROW_RIGHT = "ArrowRight",
  A_UP = "A",
  B_UP = "B",
  C_UP = "C",
  D_UP = "D",
  E_UP = "E",
  F_UP = "F",
  G_UP = "G",
  H_UP = "H",
  I_UP = "I",
  J_UP = "J",
  K_UP = "K",
  L_UP = "L",
  M_UP = "M",
  N_UP = "N",
  O_UP = "O",
  P_UP = "P",
  Q_UP = "Q",
  R_UP = "R",
  S_UP = "S",
  T_UP = "T",
  U_UP = "U",
  V_UP = "V",
  W_UP = "W",
  X_UP = "X",
  Y_UP = "Y",
  Z_UP = "Z",
  A_LOW = "a",
  B_LOW = "b",
  C_LOW = "c",
  D_LOW = "d",
  E_LOW = "e",
  F_LOW = "f",
  G_LOW = "g",
  H_LOW = "h",
  I_LOW = "i",
  J_LOW = "j",
  K_LOW = "k",
  L_LOW = "l",
  M_LOW = "m",
  N_LOW = "n",
  O_LOW = "o",
  P_LOW = "p",
  Q_LOW = "q",
  R_LOW = "r",
  S_LOW = "s",
  T_LOW = "t",
  U_LOW = "u",
  V_LOW = "v",
  W_LOW = "w",
  X_LOW = "x",
  Y_LOW = "y",
  Z_LOW = "z",
  ZERO = "0",
  ONE = "1",
  TWO = "2",
  THREE = "3",
  FOUR = "4",
  FIVE = "5",
  NINE = "9",
}

export enum TrackPolicyType {
  NONE = "none",
  LINEAR_INTERPOLATION = "linear_interpolation"
}

export enum AttributeToolType {
  SWITCH = "switch",
  LIST = "list",
  LONG_LIST = "longList",
  NONE = ""
}

export enum QueryArg {
  PROJECT_NAME = "project_name",
  TASK_INDEX = "task_index",
  TASK_ID = "task_id",
  DEV_MODE = "dev"
}
