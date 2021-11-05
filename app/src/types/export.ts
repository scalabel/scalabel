import {
  Attribute,
  ConfigType,
  ExtrinsicsType,
  IntrinsicsType,
  ItemType,
  SensorType,
  TaskStatus,
  TrackType,
  Vector3Type
} from "./state"

export interface TaskData {
  /** task config data that's constant throughout a session */
  config: ConfigType
  /** current status of task */
  status: TaskStatus
  /** items in task */
  Items: ItemType[]
  /** tracks for task */
  Tracks: TrackType
}

export interface DatasetExport {
  /** items in dataset */
  frames: ItemExport[]
  /** items group, contain multi-sensor information */
  frameGroups?: ItemGroupExport[]
  /** shared fields for frames */
  config: ConfigExport
}

export interface ItemExport {
  /** project name */
  name: string
  /** item url */
  url?: string
  /** video name */
  videoName: string
  /** id of data source */
  sensor: number
  /** data type, overrides data source if present */
  dataType?: string
  /** intrinsics, overrides data source if present */
  intrinsics?: IntrinsicsType
  /** extrinsics, overrides data source if present */
  extrinsics?: ExtrinsicsType
  /** item attributes */
  attributes: { [key: string]: string | string[] }
  /** submitted timestamp */
  timestamp: number
  /** item labels */
  labels: LabelExport[]
}

export interface ItemGroupExport extends ItemExport {
  /** name of frames, the key is the id of the corresponding sensor*/
  frames: { [id: number]: string }
}

export interface PolygonExportType {
  /** points */
  vertices: Array<[number, number]>
  /** string of types */
  types: string
  /** closed or open polygon */
  closed: boolean
}

export interface Box2DType {
  /** The x-coordinate of upper left corner */
  x1: number
  /** The y-coordinate of upper left corner */
  y1: number
  /** The x-coordinate of lower right corner */
  x2: number
  /** The y-coordinate of lower right corner */
  y2: number
}

export interface Box3DType {
  /** Center of the cube */
  center: Vector3Type
  /** size */
  size: Vector3Type
  /** orientation */
  orientation: Vector3Type
}

export interface Plane3DType {
  /** Plane origin in world */
  center: Vector3Type
  /** orientation in Euler */
  orientation: Vector3Type
}

export interface CustomExportType {
  /** point positions */
  points: Array<[number, number]>
  /** names of the points */
  names: string[]
  /** whether the points are hidden */
  hidden: boolean[]
  /** edges */
  edges: Array<[number, number]>
}

export interface LabelExport {
  /** label id */
  id: string | number
  /** label index */
  index?: number
  /** category */
  category: string
  /** label attributes- can be list or switch type */
  attributes: { [key: string]: string | string[] | boolean }
  /** if shape was manual */
  manualShape: boolean
  /** box2d label */
  box2d: Box2DType | null
  /** poly2d label */
  poly2d: PolygonExportType[] | null
  /** box3d label */
  box3d: Box3DType | null
  /** plane3d label */
  plane3d?: Plane3DType | null
  /** custom labels */
  customs?: { [name: string]: CustomExportType }
}

export interface ConfigExport {
  /** image sizes */
  image_size?: ImageSizeType
  /** attributes */
  attributes?: Attribute[]
  /** categories */
  categories: string[]
  /** sensors */
  sensors?: SensorType[]
}

export interface ImageSizeType {
  /** image width */
  width: number
  /** image height */
  height: number
}
