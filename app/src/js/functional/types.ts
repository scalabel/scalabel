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
  /** Selected shape of the label */
  selectedShape: number
  /** State */
  state: number
  /** order of the label among all the labels */
  order: number
}

export interface Track {
  /** ID of the track */
  id: number
  /** labels in this track [item index, label id] */
  labels: Array<[number, number]>
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
}

export type ShapeType = RectType | CubeType

export interface IndexedShapeType {
  /** ID of the shape */
  id: number
  /** Label ID of the shape */
  label: [number]
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
  /** View Offset X */
  viewOffsetX: number
  /** View Offset Y */
  viewOffsetY: number
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
        ImageViewerConfigType | PointCloudViewerConfigType | null

export interface ItemType {
  /** The ID of the item */
  id: number
  /** The index of the item */
  index: number
  /** The URL of the item */
  url: string
  /** Whether or not the item is active */
  active: boolean
  /** Whether or not the item is loaded */
  loaded: boolean
  /** Labels of the item */
  labels: { [key: number]: LabelType } // list of label
  /** shapes of the labels on this item */
  shapes: {[key: number]: IndexedShapeType}
  /** Configurations of the viewer */
  viewerConfig: ViewerConfigType
}

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt; put the props in config in favor
  of smaller SatProps.
 */
export interface ConfigType {
  /**
   * a unique id for each session. When the same assignment/task is opened
   * twice, they will have different session ids.
   * It is uuid of the session
   */
  sessionId: string
  /**
   * Assignment ID. The same assignment can be given to multiple sesssions
   * for collaboration
   */
  assignmentId: string
  /** Project name */
  projectName: string
  /** Item type */
  itemType: string
  /** Label types available for the session */
  labelTypes: string[]
  /** Task size */
  taskSize: number
  /** Handler URL */
  handlerUrl: string
  /** Page title */
  pageTitle: string
  /** Instruction page URL */
  instructionPage: string
  /** Whether or not in demo mode */
  demoMode: boolean
  /** Bundle file */
  bundleFile: string
  /** Categories */
  categories: string[]
  /** Attributes */
  attributes: Attribute[]
  /** Task ID */
  taskId: string
  /** Worker ID */
  workerId: string
  /** Start time */
  startTime: number
}

export interface LayoutType {
  /** Width of the tool bar */
  toolbarWidth: number
  /** Whether or not to show the assistant view */
  assistantView: boolean
  /** Assistant view ratio */
  assistantViewRatio: number
}

/*
  The current state of Sat.
 */
export interface CurrentType {
  /** Currently viewed item index */
  item: number
  /** Currently selected label ID */
  label: number
  /** Currently selected shape ID */
  shape: number
  /** selected category */
  category: number
  /** selected label type */
  labelType: number
  /** Max label ID */
  maxLabelId: number
  /** Max shape ID */
  maxShapeId: number
  /** max order number */
  maxOrder: number
}

export interface State {
  /** Configurations */
  config: ConfigType
  /** The current state */
  current: CurrentType
  /** Items */
  items: ItemType[]
  /** tracks */
  tracks: { [key: number]: Track }
  /** Layout */
  layout: LayoutType
}

export interface ProjectMetaData {
  /** project name */
  name: string
  /** item type */
  itemType: string
  /** label type */
  labelType: string
  /** task size */
  taskSize: number
  /** number of items */
  numItems: number
  /** number of categories */
  numLeafCategories: number
  /** number of attributes */
  numAttributes: number
}

export interface TaskMetaData {
  /** number of labeled images */
  numLabeledImages: string
  /** number of labels */
  numLabels: string
  /** if the task was submitted */
  submitted: boolean
  /** task link handler url */
  handlerUrl: string
}

export interface DashboardContents {
  /** project metadata */
  projectMetaData: ProjectMetaData
  /** tasks */
  taskMetaDatas: TaskMetaData[]
}

export const enum ConnectionStatus {
  SAVED, SAVING, RECONNECTING, UNSAVED
}

export type LabelFuncType =
        (id: number, itemId: number, attributes: object) => LabelType

export type ItemFuncType = (id: number, url: string) => ItemType

export interface User {
  /** User ID */
  id: string
  /** User email */
  email: string
  /** User group */
  group: string
  /** User refresh token */
  refreshToken: string
  /** User's projects */
  projects: string[]
}

export interface Attribute {
  /** Attribute tool type */
  toolType: string,
  /** Attribute name */
  name: string,
  /** Values of attribute */
  values: string[]
}
