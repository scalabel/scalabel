import { AttributeToolType } from '../common/types'

/**
 * Interfaces for immutable states
 */
export interface LabelType {
  /** ID of the label */
  id: number
  /** The item index */
  item: number
  /** type of the label */
  type: string
  /** The category ID */
  category: number[]
  /** Attributes */
  attributes: { [key: number]: number[] }
  /** Parent label ID */
  parent: number
  /** Children label IDs */
  children: number[]
  /** Shape ids of the label */
  shapes: number[]
  /** connected track */
  track: number
  /** order of the label among all the labels */
  order: number
  /** whether the label is created manually */
  manual: boolean
}

export interface TrackType {
  /** ID of the track */
  id: number
  /** labels in this track {item index: label id} */
  labels: {[key: number]: number}
}

export interface RectType {
  /** The x-coordinate of upper left corner */
  x1: number
  /** The y-coordinate of upper left corner */
  y1: number
  /** The x-coordinate of lower right corner */
  x2: number
  /** The y-coordinate of lower right corner */
  y2: number
}

export interface PolygonType {
  /** array of control points */
  points: PathPoint2DType []
}

export interface Vector3Type {
  /** The x-coordinate */
  x: number
  /** The y-coordinate */
  y: number
  /** The z-coordinate */
  z: number
}

export interface CubeType {
  /** Center of the cube */
  center: Vector3Type
  /** size */
  size: Vector3Type
  /** orientation */
  orientation: Vector3Type
  /** Anchor corner index for reshaping */
  anchorIndex: number
  /** ID of the surface this box is attached to */
  surfaceId: number
}

export interface Point2DType {
  /** The x-coordinate */
  x: number
  /** The y-coordinate */
  y: number
}

export interface PathPoint2DType {
  /** The x-coordinate */
  x: number
  /** The y-coordinate */
  y: number
  /** type of the point in the path. value from common/types.PathPointType */
  type: string
}

export interface Plane3DType {
  /** Plane origin in world */
  center: Vector3Type
  /** orientation in Euler */
  orientation: Vector3Type
}

export type ShapeType = RectType | CubeType | PolygonType |
                        Point2DType | PathPoint2DType | Plane3DType

export interface IndexedShapeType {
  /** ID of the shape */
  id: number
  /** Label ID of the shape */
  label: number[]
  /** type string of the shape. Value from common/types.ShapeType */
  type: string
  /** Shape data */
  shape: ShapeType
}

export interface ImageViewerConfigType {
  /** The width of the image */
  imageWidth: number
  /** The height of the image */
  imageHeight: number
  /** View scale */
  viewScale: number
  /** Display Scroll Top */
  displayTop: number
  /** Display Scroll Left */
  displayLeft: number
}

export interface PointCloudViewerConfigType {
  /** Camera position */
  position: Vector3Type
  /** Viewing direction */
  target: Vector3Type
  /** Up direction of the camera */
  verticalAxis: Vector3Type
}

export type ViewerConfigType =
  ImageViewerConfigType | PointCloudViewerConfigType

export interface ItemType {
  /** The ID of the item */
  id: number
  /** The index of the item */
  index: number
  /** The URL of the item */
  url: string
  /** Labels of the item */
  labels: { [key: number]: LabelType } // list of label
  /** shapes of the labels on this item */
  shapes: { [key: number]: IndexedShapeType }
  /** the timestamp for the item */
  timestamp: number
}

export interface Attribute {
  /** Attribute tool type */
  toolType: AttributeToolType,
  /** Attribute name */
  name: string,
  /** Values of attribute */
  values: string[],
  /** Tag text */
  tagText: string,
  /** Tag prefix */
  tagPrefix: string,
  /** Tag suffixes */
  tagSuffixes: string[]
  /** button colors */
  buttonColors: string[]
}

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt; put the props in config in favor
  of smaller SatProps.
 */
export interface ConfigType {
  /** Project name */
  projectName: string
  /** Item type */
  itemType: string
  /** Label types available for the session */
  labelTypes: string[]
  /** Policy types available for session */
  policyTypes: string[]
  /** Task size */
  taskSize: number
  /** Whether to track */
  tracking: boolean
  /** Handler URL */
  handlerUrl: string
  /** Page title */
  pageTitle: string
  /** Instruction page URL */
  instructionPage: string
  /** Bundle file */
  bundleFile: string
  /** Categories */
  categories: string[]
  /** Attributes */
  attributes: Attribute[]
  /** task id */
  taskId: string
  /** the time of last project submission */
  submitTime: number
  /** Whether or not in demo mode */
  demoMode: boolean
  /** Whether or not submitted */
  submitted: boolean
}

export interface LayoutType {
  /** Width of the tool bar */
  toolbarWidth: number
  /** Whether or not to show the assistant view */
  assistantView: boolean
  /** Assistant view ratio */
  assistantViewRatio: number
}

export interface TaskStatus {
  /** Max label ID */
  maxLabelId: number
  /** Max shape ID */
  maxShapeId: number
  /** max order number */
  maxOrder: number
  /** max track ID */
  maxTrackId: number
}

export interface TrackMapType { [key: number]: TrackType }

export interface TaskType {
  /** Configurations */
  config: ConfigType
  /** The current state */
  status: TaskStatus
  /** Items */
  items: ItemType[]
  /** tracks */
  tracks: TrackMapType
}

export interface Select {
  /** Currently viewed item index */
  item: number
  /** Currently selected label ID */
  labels: number[]
  /** Currently selected shape ID */
  shapes: number[]
  /** selected category */
  category: number
  /** selected attributes */
  attributes: {[key: number]: number[]}
  /** selected label type */
  labelType: number
  /** selected track policy type */
  policyType: number
}

/**
 * User information that may persist across sessions
 */
export interface UserType {
  /** user id. the worker can be a guest or registered user */
  id: string
  /** the selection of the current user */
  select: Select
  /** interface layout */
  layout: LayoutType
  /** Viewer configuration for images */
  imageViewerConfig: ImageViewerConfigType
  /** Viewer configuration for point clouds */
  pointCloudViewerConfig: PointCloudViewerConfigType
}

export interface ItemStatus {
  /** whether this item is loaded in this session */
  loaded: boolean
}

/**
 * Information for this particular session
 */
export interface SessionType {
  /**
   * a unique id for each session. When the same assignment/task is opened
   * twice, they will have different session ids.
   * It is uuid of the session
   */
  id: string
  /** Start time */
  startTime: number
  /** item statuses */
  items: ItemStatus[]
}

export interface State {
  /**
   * task config and labels. It is irrelevant who makes the labels and other
   * content in task
   */
  task: TaskType
  /** user information that can be persistent across sessions */
  user: UserType
  /** info particular to this session */
  session: SessionType
}
