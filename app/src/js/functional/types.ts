import { AttributeToolType } from '../common/types'

/**
 * Interfaces for immutable states
 */
export interface LabelType {
  /** ID of the label */
  id: number
  /** The item index */
  item: number
  /** Associated data sources */
  sensors: number[]
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

export interface Vector2Type {
  /** The x-coordinate */
  x: number
  /** The y-coordinate */
  y: number
}

export interface Vector3Type {
  /** The x-coordinate */
  x: number
  /** The y-coordinate */
  y: number
  /** The z-coordinate */
  z: number
}

export interface Vector4Type {
  /** The w-coordinate */
  w: number
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

export type Point2DType = Vector2Type

export interface PathPoint2DType extends Point2DType {
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

export interface ViewerConfigType {
  /** string indicating type */
  type: string
  /** whether to show */
  show: boolean
  /** which data sources to view */
  sensor: number
  /** id of pane this belongs to */
  pane: number
  /**
   * Set if synchronized with compatible viewer configs,
   * must be same type for now
   */
  synchronized: boolean
  /** whether to hide non-selected labels */
  hideLabels: boolean
}

export interface ImageViewerConfigType extends ViewerConfigType {
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

export interface PointCloudViewerConfigType extends ViewerConfigType {
  /** Camera position */
  position: Vector3Type
  /** Viewing direction */
  target: Vector3Type
  /** Up direction of the camera */
  verticalAxis: Vector3Type
  /** Camera rotation lock */
  lockStatus: number
}

export interface Image3DViewerConfigType extends
  ImageViewerConfigType {
  /** If set, sensor id of point cloud to use as reference */
  pointCloudSensor: number
}

export interface CameraIntrinsicsType {
  /** focal length 2d */
  focalLength: Vector2Type
  /** focal center 2d */
  focalCenter: Vector2Type
}

export type IntrinsicsType = CameraIntrinsicsType

export interface ExtrinsicsType {
  /** rotation to data source frame */
  rotation: Vector4Type
  /** translation to data source frame */
  translation: Vector3Type
}

export interface SensorType {
  /** id */
  id: number
  /** name */
  name: string
  /** data type */
  type: string
  /** intrinsics */
  intrinsics?: IntrinsicsType
  /** extrinsics */
  extrinsics?: ExtrinsicsType
}

export interface SensorMapType { [id: number]: SensorType }

export interface ItemType {
  /** The ID of the item */
  id: number
  /** The index of the item */
  index: number
  /** Map between data source id and url */
  urls: {[id: number]: string}
  /** Labels of the item */
  labels: { [key: number]: LabelType } // list of label
  /** shapes of the labels on this item */
  shapes: { [key: number]: IndexedShapeType }
  /** the timestamp for the item */
  timestamp: number
  /** video item belongs to */
  videoName: string
}

export interface Node2DType extends Point2DType {
  /** name */
  name: string
  /** color */
  color?: number[]
  /** set if hidden */
  hidden?: boolean
}

// TODO: This only supports points for now.
// Needs to be extended to support polygons as basic part type
export interface Label2DTemplateType {
  /** spec name */
  name: string
  /** template */
  nodes: Node2DType[]
  /** connections between points represented as array of 2d tuples */
  edges: Array<[number, number]>
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
  /** item type */
  itemType: string
  /** Label types available for the session */
  labelTypes: string[]
  /** Custom label template */
  label2DTemplates: { [name: string]: Label2DTemplateType }
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
  /** whether to use autosave */
  autosave: boolean
}

export enum SplitType {
  HORIZONTAL = 'horizontal',
  VERTICAL = 'vertical'
}

export interface PaneType {
  /** id of the pane */
  id: number
  /** id of parent pane, negative for root */
  parent: number
  /** If leaf, >= 0 */
  viewerId: number
  /**
   * Which child is the primary pane to apply sizing to.
   * Other child is sized based on the size of the primary child
   * (100% - primary width)
   */
  primary?: 'first' | 'second'
  /** Size of primary pane. */
  primarySize?: number | string
  /** Split type, horizontal or vertical */
  split?: SplitType
  /** Min size of primary pane */
  minPrimarySize?: number
  /** Max size of primary pane */
  maxPrimarySize?: number
  /** Id of first child if not leaf */
  child1?: number
  /** Id of second child if not leaf */
  child2?: number
}

export interface LayoutType {
  /** Width of the tool bar */
  toolbarWidth: number
  /** max viewer config id */
  maxViewerConfigId: number
  /** max viewer config id */
  maxPaneId: number
  /** top level pane node */
  rootPane: number
  /** map between pane id and pane states */
  panes: {[id: number]: PaneType}
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
  /** data sources */
  sensors: SensorMapType
}

export interface Select {
  /** Currently viewed item index */
  item: number
  /** Map between item indices and label id's */
  labels: {[index: number]: number[]}
  /** Map between label id's and shape id's */
  shapes: {[index: number]: number}
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
  /** Viewer configurations, only id 0 & 1 for now (main & assistant) */
  viewerConfigs: {[id: number]: ViewerConfigType}
}

export interface ItemStatus {
  /** Whether data source in item is loaded */
  sensorDataLoaded: {[id: number]: boolean}
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
  itemStatuses: ItemStatus[]
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
