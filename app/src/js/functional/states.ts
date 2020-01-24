import _ from 'lodash'
import * as types from '../common/types'
import {
  ConfigType, CubeType,
  ExtrinsicsType, Image3DViewerConfigType,
  ImageViewerConfigType, IndexedShapeType, IntrinsicsType, ItemStatus,
  ItemType,
  LabelType,
  LayoutType,
  PaneType,
  PathPoint2DType,
  Plane3DType,
  PointCloudViewerConfigType,
  PolygonType,
  RectType,
  Select,
  SensorType,
  SessionType,
  ShapeType,
  SplitType,
  State,
  TaskStatus,
  TaskType,
  TrackType,
  UserType,
  ViewerConfigType
} from './types'

/**
 * Initialize a label state and deep copy the parameters
 * @param {Partial<LabelType>} params
 * @return {LabelType}
 */
export function makeLabel (params: Partial<LabelType> = {}): LabelType {
  return _.cloneDeep<LabelType>({
    id: -1,
    item: -1,
    sensors: [-1],
    type: types.LabelTypeName.EMPTY,
    category: [],
    attributes: {},
    parent: -1, // id
    children: [], // ids
    shapes: [],
    track: -1,
    order: 0,
    manual: true, // by default, manual is true
    ...params
  })
}

/**
 * Initialize a track
 * @param {number} id
 * @param {{[key: number]: number}} labels
 */
export function makeTrack (
  id: number, labels: {[key: number]: number} = {}
): TrackType {
  return { id, labels }
}

/**
 * Initialize a rectangle shape
 * @param {{}} params
 * @return {RectType}
 */
export function makeRect (params: Partial<RectType> = {}): RectType {
  return {
    x1: -1,
    y1: -1,
    x2: -1,
    y2: -1,
    ...params
  }
}

/**
 * Initialize a polygon
 * @param {{}} params
 * @return {PolygonType}
 */
export function makePolygon
  (params: Partial<PolygonType> = {}): PolygonType {
  return {
    points: [],
    ...params
  }
}

/**
 * Initialize a pathPoint shape
 * @param params
 */
export function makePathPoint (params: Partial<PathPoint2DType> = {})
: PathPoint2DType {
  return {
    x: 0,
    y: 0,
    type: 'vertex',
    ...params
  }
}

/**
 * Initialize a 3d box shape
 * @param {{}} params
 * @return {CubeType}
 */
export function makeCube (params: Partial<CubeType> = {}): CubeType {
  return {
    center: { x: 0, y: 0, z: 0 },
    size: { x: 1, y: 1, z: 1 },
    orientation: { x: 0, y: 0, z: 0 },
    anchorIndex: 0,
    surfaceId: -1,
    ...params
  }
}

/**
 * Initialize a 3d box shape
 * @param {{}} params
 * @return {Plane3DType}
 */
export function makePlane (params: {} = {}): Plane3DType {
  return {
    center: { x: 0, y: 0, z: 0 },
    orientation: { x: 0, y: 0, z: 0 },
    ...params
  }
}

/**
 * Compose indexed shape
 * @param {number} id
 * @param {number[]} label
 * @param {string} type
 * @param {ShapeType} shape
 */
export function makeIndexedShape (
    id: number, label: number[], type: string, shape: ShapeType
  ): IndexedShapeType {
  return {
    id, label: [...label], type, shape: { ...shape }
  }
}

/**
 * Create data source
 * @param id
 * @param type
 * @param name
 */
export function makeSensor (
  id: number,
  name: string,
  type: string,
  intrinsics?: IntrinsicsType,
  extrinsics?: ExtrinsicsType
): SensorType {
  return { id, name, type, intrinsics, extrinsics }
}

/**
 * Make a new viewer config
 * @return {ImageViewerConfigType}
 */
export function makeImageViewerConfig (
  pane: number, sensor: number = -1
): ImageViewerConfigType {
  return {
    imageWidth: 0,
    imageHeight: 0,
    viewScale: 1.0,
    displayLeft: 0,
    displayTop: 0,
    show: true,
    type: types.ViewerConfigTypeName.IMAGE,
    sensor,
    pane,
    synchronized: false,
    hideLabels: false
  }
}

/**
 * Make a new point cloud viewer config
 * @return {PointCloudViewerConfigType}
 */
export function makePointCloudViewerConfig (
  pane: number, sensor: number = -1
): PointCloudViewerConfigType {
  return {
    position: { x: 0.0, y: 0.0, z: 0.0 },
    target: { x: 10.0, y: 0.0, z: 0.0 },
    verticalAxis: { x: 0.0, y: 0.0, z: 1.0 },
    lockStatus: 0,
    show: true,
    type: types.ViewerConfigTypeName.POINT_CLOUD,
    sensor,
    pane,
    synchronized: false,
    hideLabels: false
  }
}

/**
 * Make image 3d viewer config
 * @param pane
 * @param sensor
 */
export function makeImage3DViewerConfig (
  pane: number, sensor: number = -1
): Image3DViewerConfigType {
  const imageConfig = makeImageViewerConfig(pane, sensor)
  return {
    ...imageConfig,
    type: types.ViewerConfigTypeName.IMAGE_3D,
    pointCloudSensor: -2
  }
}

/**
 * Create viewer config based on type
 * @param type
 * @param pane
 * @param sensor
 */
export function makeDefaultViewerConfig (
  type: types.ViewerConfigTypeName,
  pane: number,
  sensor: number
): ViewerConfigType | null {
  switch (type) {
    case types.ViewerConfigTypeName.IMAGE:
      return makeImageViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.IMAGE_3D:
      return makeImage3DViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return makePointCloudViewerConfig(pane, sensor)
  }
  return null
}

/**
 * Initialize an item state
 * @param {{}} params
 * @return {ItemType}
 */
export function makeItem (params: Partial<ItemType> = {}): ItemType {
  return {
    id: -1,
    index: 0,
    videoName: '',
    urls: {},
    labels: {},
    shapes: {},
    timestamp: -1,
    ...params
  }
}

/**
 * Make Sat configuration state
 * @param {{}} params
 * @return {ConfigType}
 */
export function makeTaskConfig (params: Partial<ConfigType> = {}): ConfigType {
  return {
    projectName: '',
    itemType: '',
    labelTypes: [],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: 0,
    tracking: false,
    handlerUrl: '',
    pageTitle: '',
    instructionPage: '', // instruction url
    bundleFile: '',
    categories: [],
    attributes: [],
    taskId: '',
    submitTime: 0,
    demoMode: false,
    submitted: false,
    autosave: false,
    ...params
  }
}

/**
 * Make pane node
 * @param viewerId
 * @param size
 * @param split
 * @param minSize
 * @param maxSize
 */
export function makePane (
  viewerId: number = -1,
  paneId: number = -1,
  parent: number = -1,
  primarySize?: number,
  split?: SplitType,
  primary?: 'first' | 'second',
  minPrimarySize?: number,
  maxPrimarySize?: number,
  child1?: number,
  child2?: number
): PaneType {
  return {
    id: paneId,
    viewerId,
    parent,
    primary,
    primarySize,
    split,
    minPrimarySize,
    maxPrimarySize,
    child1,
    child2
  }
}

/**
 * Initialize a Layout state
 * @param {{}} params
 * @return {LayoutType}
 */
export function makeLayout (params: {} = {}): LayoutType {
  return {
    toolbarWidth: 200,
    maxViewerConfigId: 0,
    maxPaneId: 0,
    rootPane: 0,
    panes: { [0]: makePane(0, 0) },
    ...params
  }
}

/**
 * Initialize a user selection sate
 * @param {{}} params
 * @return {Selection}
 */
function makeSelect (params: Partial<Select>= {}): Select {
  return {
    item: 0,
    labels: [],
    shapes: [],
    category: 0,
    attributes: {},
    labelType: 0,
    policyType: 0,
    ...params
  }
}

/**
 * Initialize a user sate
 * @param {{}} params
 * @return {UserType}
 */
function makeUser (params: Partial<UserType>= {}): UserType {
  return {
    id: '',
    select: makeSelect(),
    layout: makeLayout(),
    viewerConfigs: [],
    ...params
  }
}

/**
 * Initialize a item status sate
 * @param {{}} params
 * @return {ItemStatus}
 */
export function makeItemStatus (params: Partial<ItemStatus>= {}): ItemStatus {
  return {
    sensorDataLoaded: {},
    ...params
  }
}

/**
 * Initialize a session state
 * @param {{}} params
 * @return {Session}
 */
function makeSession (params: Partial<SessionType>= {}): SessionType {
  return {
    id: '',
    startTime: 0,
    itemStatuses: [],
    ...params
  }
}

/**
 * Initialize a task status state
 * @param {{}} params
 * @return {TaskStatus}
 */
function makeTaskStatus (params: Partial<TaskStatus> = {}): TaskStatus {
  return {
    maxLabelId: -1,
    maxShapeId: -1,
    maxOrder: -1,
    maxTrackId: -1,
    ...params
  }
}

/**
 * Initialize a task sate
 * @param {{}} params
 * @return {TaskType}
 */
export function makeTask (params: Partial<TaskType> = {}): TaskType {
  return {
    config: makeTaskConfig(),
    status: makeTaskStatus(),
    items: [],
    tracks: {},
    sensors: {},
    ...params
  }
}

/**
 * Initialize a Sat state
 * @param {{}} params
 * @return {State}
 */
export function makeState (params: Partial<State> = {}): State {
  return {
    task: makeTask(params.task),
    user: makeUser(params.user),
    session: makeSession(params.session)
  }
}
