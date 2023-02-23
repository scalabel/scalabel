import { AttributeToolType } from "../const/common"
import { Span3D } from "../drawable/3d/box_span/span3d"

export type IdType = string
export const INVALID_ID: IdType = ""

// Have to define those map because
// we can't use alias of string as an index signature parameter type
export interface LabelIdMap {
  [key: string]: LabelType
}

export interface ShapeIdMap {
  [key: string]: ShapeAllType
}

export interface TrackIdMap {
  [key: string]: TrackType
}

/**
 * Interfaces for immutable states
 */
export interface LabelType {
  /** ID of the label */
  id: IdType
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
  parent: IdType
  /** Children label IDs */
  children: IdType[]
  /** Shape ids of the label */
  shapes: IdType[]
  /** connected track */
  track: IdType
  /** order of the label among all the labels */
  order: number
  /** whether the label is created manually */
  manual: boolean
  /** Has shape changed? e.g. vertex is added/deleted. */
  changed: boolean
  /** whether the label has been annotated as checked. */
  checked: boolean
}

export interface TrackType {
  /** ID of the track */
  id: IdType
  /** type */
  type: string
  /** labels in this track {item index: label id} */
  labels: { [key: number]: IdType }
}

export interface ShapeType {
  /** ID of the shape */
  id: IdType
  /** Label ID of the shape */
  label: IdType[]
  /** type string of the shape. Value from common/types.ShapeType */
  shapeType: string
}

export interface SimpleRect {
  /** The x-coordinate of upper left corner */
  x1: number
  /** The y-coordinate of upper left corner */
  y1: number
  /** The x-coordinate of lower right corner */
  x2: number
  /** The y-coordinate of lower right corner */
  y2: number
}

export interface RectType extends ShapeType, SimpleRect {}

export interface PolygonType extends ShapeType {
  /** array of control points */
  points: PolyPathPoint2DType[]
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

export interface SimpleCube {
  /** Center of the cube */
  center: Vector3Type
  /** size */
  size: Vector3Type
  /** orientation */
  orientation: Vector3Type
  /** Anchor corner index for reshaping */
  anchorIndex: number
}

export interface CubeType extends ShapeType, SimpleCube {}

export type Point2DType = Vector2Type

export enum PathPointType {
  UNKNOWN = "null",
  LINE = "line",
  CURVE = "bezier",
  MID = "mid"
}

export interface SimplePathPoint2DType extends Vector2Type {
  /** type of the point in the path. */
  pointType: PathPointType
}

export interface PathPoint2DType extends SimplePathPoint2DType, ShapeType {}

/**
 * This is a point on the polygon 2d. So the point itself doesn't have an ID
 * and it is not standalone shape
 */
export interface PolyPathPoint2DType extends Vector2Type {
  /** type of the point in the path. value from common/types.PathPointType */
  pointType: string
}

export interface Plane3DType extends ShapeType {
  /** Plane origin in world */
  center: Vector3Type
  /** orientation in Euler */
  orientation: Vector3Type
}

export type ShapeAllType =
  | ShapeType
  | RectType
  | CubeType
  | PolygonType
  | Plane3DType
  | Node2DType

export interface IndexedShapeType {
  /** ID of the shape */
  id: IdType
  /** Label ID of the shape */
  labels: IdType[]
  /** type string of the shape. Value from common/types.ShapeType */
  shapeType: string
  /** Shape data */
  shape: ShapeType
}

export enum ColorSchemeType {
  IMAGE = "image",
  DEPTH = "depth",
  HEIGHT = "height"
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
  /** whether to hide label tags */
  hideTags: boolean
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
  /** Whether to flip the direction of the axis locks */
  flipAxis: boolean
  /** Camera rotation lock */
  lockStatus: number
  /** Camera rotation direction */
  cameraRotateDir?: boolean
  /** Color scheme for the point cloud */
  colorScheme: ColorSchemeType
  /** Has the camera been transformed */
  cameraTransformed: boolean
}

export interface Image3DViewerConfigType extends ImageViewerConfigType {
  /** If set, sensor id of point cloud to use as reference */
  pointCloudSensor: number
  /** whether to overlay point cloud */
  pointCloudOverlay: boolean
  /** Viewing direction */
  target: Vector3Type
  /** Up direction of the camera */
  verticalAxis: Vector3Type
}

export interface HomographyViewerConfigType extends Image3DViewerConfigType {
  /** Distance to plane */
  distance: number
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
  /** sensor height */
  height?: number
  /** radar */
  radar?: string
}

export interface SensorMapType {
  [id: number]: SensorType
}

export interface ItemType {
  /** project name */
  names?: { [id: number]: string }
  /** The ID of the item */
  id: IdType
  /** The index of the item */
  index: number
  /** Map between data source id and url */
  urls: { [id: number]: string }
  /** Labels of the item */
  labels: LabelIdMap // List of label
  /** shapes of the labels on this item */
  shapes: ShapeIdMap
  /** the timestamp for the item */
  timestamp: number
  /** video item belongs to */
  videoName: string
  /** intrinsics, overrides data source if present */
  intrinsics?: { [id: number]: IntrinsicsType }
  /** extrinsics, overrides data source if present */
  extrinsics?: { [id: number]: ExtrinsicsType }
}

export interface Node2DType extends Vector2Type, ShapeType {
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

export interface Category {
  /** Category name */
  name: string
  /** Children categories if tree structure */
  subcategories?: Category[]
}

export interface Attribute {
  /** Attribute tool type */
  type: AttributeToolType
  /** Attribute name */
  name: string
  /** Values of attribute */
  values: string[]
  /** Tag text */
  tag: string
  /** Tag prefix */
  tagPrefix: string
  /** Tag suffixes */
  tagSuffixes: string[]
  /** button colors */
  buttonColors?: string[]
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
  /** Keyframe interval */
  keyInterval: number
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
  /** Categories in tree structure*/
  treeCategories: Category[]
  /** Attributes */
  attributes: Attribute[]
  /** task id */
  taskId: IdType
  /** Whether or not in demo mode */
  demoMode: boolean
  /** whether to use autosave */
  autosave: boolean
  /** whether bots are enabled */
  bots: boolean
}

export enum SplitType {
  HORIZONTAL = "horizontal",
  VERTICAL = "vertical"
}

export interface PaneType {
  /** id of the pane */
  id: number
  /** id of parent pane, negative for root */
  parent: number
  /** If leaf, >= 0 */
  viewerId: number
  /** Whether to hide pane */
  hide: boolean
  /**
   * Which child is the primary pane to apply sizing to.
   * Other child is sized based on the size of the primary child
   * (100% - primary width)
   */
  primary?: "first" | "second"
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
  /** Number of children that are vertical splits */
  numVerticalChildren: number
  /** Number of children that are horizontal splits */
  numHorizontalChildren: number
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
  panes: { [id: number]: PaneType }
}

export interface TaskStatus {
  /** max order number */
  maxOrder: number
}

export interface SubmitData {
  /** time of the submission (client side) */
  time: number
  /** author of the submission */
  user: string
}

export interface Progress {
  /** data on submissions */
  submissions: SubmitData[]
}
export interface TaskType {
  /** Configurations */
  config: ConfigType
  /** The current state */
  status: TaskStatus
  /** Items */
  items: ItemType[]
  /** tracks */
  tracks: TrackIdMap
  /** data sources */
  sensors: SensorMapType
  /** info on task progress */
  progress: Progress
}

export interface Select {
  /** Currently viewed item index */
  item: number
  /** Map between item indices and label id's */
  labels: { [index: number]: IdType[] }
  /** Map between label id's and shape id's */
  shapes: { [index: string]: IdType }
  /** selected category */
  category: number
  /** selected attributes */
  attributes: { [key: number]: number[] }
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
  id: IdType
  /** the selection of the current user */
  select: Select
  /** interface layout */
  layout: LayoutType
  /** Viewer configurations, only id 0 & 1 for now (main & assistant) */
  viewerConfigs: { [id: number]: ViewerConfigType }
}

export interface ItemStatus {
  /** Whether data source in item is loaded */
  sensorDataLoaded: { [id: number]: boolean }
}

// export interface OverlayStatus{
//   /** Whether an overlay is currently loaded */
//   overlayDisplayed: { [id: number]: boolean }
// }

export const enum ConnectionStatus {
  NOTIFY_SAVED,
  SAVED,
  SAVING,
  RECONNECTING,
  UNSAVED,
  COMPUTING,
  COMPUTE_DONE,
  NOTIFY_COMPUTE_DONE,
  SUBMITTING,
  SUBMITTED,
  NOTIFY_SUBMITTED
}

export const enum ModeStatus {
  ANNOTATING,
  SELECTING
}

export interface AlertType {
  /** alert id */
  id: string
  /** alert message */
  message: string
  /** alert severity */
  severity: string
  /** alert timeout */
  timeout: number
}

export interface Info3DType {
  /** Box span toggled */
  isBoxSpan: boolean
  /** Bounding box created by spanning mode */
  boxSpan: Span3D | null
  /** Ground plane toggled */
  showGroundPlane: boolean
}

export interface Polygon2DBoundaryCloneStatusType {
  /** Id of ther label to be clone from */
  labelId: IdType | undefined
  /** Index of the first handler */
  handler1Idx: number | undefined
  /** Index of the second handler */
  handler2Idx: number | undefined
  /** If to clone the other way around */
  reverse: boolean
}

/**
 * Information for this particular session
 */
export interface SessionType {
  /**
   * a unique id for each session. When the same assignment/task is opened
   * twice, they will have different session ids.
   * It is uid of the session
   */
  id: IdType
  /** Start time */
  startTime: number
  /** Item statuses */
  itemStatuses: ItemStatus[]
  /** Track linking toggled */
  trackLinking: boolean
  /** Polygon2D boundary clone status */
  polygon2DBoundaryClone: Polygon2DBoundaryCloneStatusType | undefined
  /** Current connection status */
  status: ConnectionStatus
  /** Current mode */
  mode: ModeStatus
  /** Number of time status has changed */
  numUpdates: number
  /** Alerts */
  alerts: AlertType[]
  /** 3D-specific information */
  info3D: Info3DType
  /** Active Overlays */
  overlayStatus: number[] 
  /** Overlay Transperancy */
  overlayTransparency: number
  /** Radar Status, [x,y,value] */
  radarStatus: number[]
}

export interface State {
  /**
   * Task config and labels. It is irrelevant who makes the labels and other
   * content in task
   */
  task: TaskType
  /** User information that can be persistent across sessions */
  user: UserType
  /** Info particular to this session */
  session: SessionType
}

// Make properties optional at all levels of nested object
type RecursivePartial<T> = {
  [P in keyof T]?: RecursivePartial<T[P]>
}

/**
 * State where all properties are optional
 * Used for initialization with test objects
 */
export type DeepPartialState = RecursivePartial<State>
