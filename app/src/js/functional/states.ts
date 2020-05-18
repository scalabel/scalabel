import { bool } from 'aws-sdk/clients/signer'
import _ from 'lodash'
import * as types from '../common/types'
import { uid } from '../common/uid'
import { ItemExport, LabelExport } from '../server/bdd_types'
import { taskIdToString } from './id2string'
import {
  ConfigType, ConnectionStatus, CubeType,
  ExtrinsicsType, HomographyViewerConfigType,
  IdType,
  Image3DViewerConfigType,
  ImageViewerConfigType,
  IntrinsicsType,
  ItemStatus,
  ItemType,
  LabelType,
  LayoutType,
  Node2DType,
  PaneType,
  PathPoint2DType,
  Plane3DType,
  PointCloudViewerConfigType,
  PolygonType,
  PolyPathPoint2DType,
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
 * Initialize a label state and deep copy the parameters.
 * Every label has an id when it is born.
 * @param {Partial<LabelType>} params
 * @param {boolean} keepId whether to keep the ID in params
 * @return {LabelType}
 */
export function makeLabel (params: Partial<LabelType> = {},
                           keepId: boolean = false): LabelType {
  const label: LabelType = {
    id: genLabelId(),
    item: -1,
    sensors: [-1],
    type: types.LabelTypeName.EMPTY,
    category: [],
    attributes: {},
    parent: makeDefaultId(), // id
    children: [], // ids
    shapes: [],
    track: makeDefaultId(),
    order: 0,
    manual: true, // by default, manual is true
    ...params
  }
  if (!keepId) {
    label.id = genLabelId()
  }
  return label
}

/**
 * Initialize a track
 * Every track has an id when it is born
 * @param {Partial<TrackType>} params
 * @param {boolean} keepId whether to keep the ID in params
 * Labels can not be filled without specifying id
 */
export function makeTrack (params: Partial<TrackType> = {},
                           keepId: boolean = false): TrackType {
  const track: TrackType = {
    id: genTrackId(),
    type: types.LabelTypeName.EMPTY,
    labels: {},
    ...params
  }
  if (!keepId) {
    track.id = genTrackId()
  }
  return track
}

/**
 * Make an empty shape
 * Every shape has an id when it is born
 * @param {string} shapeType type name of the shape
 */
function makeShape (shapeType: string = '',
                    params: Partial<ShapeType> = {}): ShapeType {
  return {
    label: [],
    shapeType,
    ...params,
    id: genShapeId()
  }
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
    ...params,
    ...makeShape(types.ShapeTypeName.RECT)
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
    ...params,
    ...makeShape(types.ShapeTypeName.POLYGON_2D)
  }
}

/**
 * Initialize a polygon
 * @param {{}} params
 * @return {PolygonType}
 */
export function makePathPoint2D (
    params: Partial<PathPoint2DType> = {}): PathPoint2DType {
  return {
    x: -1,
    y: -1,
    pointType: 'vertex',
    ...params,
    ...makeShape(types.ShapeTypeName.PATH_POINT_2D)
  }
}

/**
 * Initialize a pathPoint shape
 * @param params
 */
export function makePolyPathPoint (params: Partial<PolyPathPoint2DType> = {})
  : PolyPathPoint2DType {
  return {
    x: -1,
    y: -1,
    pointType: 'vertex',
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
    ...params,
    ...makeShape(types.ShapeTypeName.CUBE)
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
    ...params,
    ...makeShape(types.ShapeTypeName.GRID)
  }
}

/**
 * Make a Node2D type
 * @param params
 */
export function makeNode2d (params: Partial<Node2DType> = {}): Node2DType {
  return {
    name: '',
    hidden: false,
    x: -1,
    y: -1,
    ...params,
    ...makeShape(types.ShapeTypeName.NODE_2D)
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
    flipAxis: false,
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
 * Make homography viewer config
 * @param pane
 * @param sensor
 */
export function makeHomographyViewerConfig (
  pane: number, sensor: number = -1, distance: number = 10
): HomographyViewerConfigType {
  const imageConfig = makeImageViewerConfig(pane, sensor)
  return {
    ...imageConfig,
    type: types.ViewerConfigTypeName.HOMOGRAPHY,
    pointCloudSensor: -2,
    distance
  }
}

/**
 * Create default viewer config for item type
 * @param sensors
 * @param type
 * @param pane
 * @param sensor
 */
export function makeDefaultViewerConfig (
  type: types.ViewerConfigTypeName, pane: number = 0, sensor: number = -1
): ViewerConfigType | null {
  switch (type) {
    case types.ViewerConfigTypeName.IMAGE:
      return makeImageViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.IMAGE_3D:
      return makeImage3DViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return makePointCloudViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.HOMOGRAPHY:
      return makeHomographyViewerConfig(pane, sensor)
  }
  return null
}

/**
 * Initialize an item state
 * @param {{}} params
 * @param {boolean} keepId Use the input param id instead of generating a new ID
 * @return {ItemType}
 */
export function makeItem (params: Partial<ItemType> = {},
                          keepId: boolean = false): ItemType {
  const item: ItemType = {
    id: genItemId(),
    index: 0,
    videoName: '',
    urls: {},
    labels: {},
    shapes: {},
    timestamp: -1,
    ...params
  }
  if (!keepId) {
    item.id = genItemId()
  }
  return item
}

/**
 * Initialize an exportable item
 * @param {{}} params
 * @return {ItemExport}
 */
export function makeItemExport (params: Partial<ItemExport> = {}): ItemExport {
  return {
    name: '',
    url: '',
    videoName: '',
    sensor: -1,
    attributes: {},
    timestamp: -1,
    labels: [],
    ...params
  }
}

/**
 * Initialize an exportable label
 * @param {{}} params
 * @return {ItemExport}
 */
export function makeLabelExport (
  params: Partial<LabelExport> = {}): LabelExport {
  return {
    id: makeDefaultId(),
    category: '',
    attributes: {},
    manualShape: true,
    box2d: null,
    poly2d: null,
    box3d: null,
    plane3d: null,
    customs: {},
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
    demoMode: false,
    autosave: false,
    bots: false,
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
    child2,
    hide: false,
    numHorizontalChildren: 0,
    numVerticalChildren: 0
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
    shapes: {},
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
    id: makeDefaultId(),
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
    id: makeDefaultId(),
    startTime: 0,
    itemStatuses: [],
    trackLinking: false,
    status: ConnectionStatus.UNSAVED,
    numUpdates: 0,
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
    maxOrder: -1,
    ...params
  }
}

/**
 * Initialize a task sate
 * @param {{}} params
 * @return {TaskType}
 */
export function makeTask (params: Partial<TaskType> = {}): TaskType {
  const task: TaskType = {
    config: makeTaskConfig(),
    status: makeTaskStatus(),
    items: [],
    tracks: {},
    sensors: {},
    progress: {
      submissions: []
    },
    ...params
  }
  return taskIdToString(task)
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

/**
 * Check whether the input ID is valid or not default
 * @param {IdType} id
 */
export function isValidId (id: IdType): bool {
  return id !== '' && id !== '-1'
}

/**
 * Make default ID
 */
export function makeDefaultId (): IdType {
  return ''
}

/**
 * Generate new label id. It should not be called outside this file.
 */
export function genLabelId (): IdType {
  return uid()
}

/**
 * Generate new track id. It should not be called outside this file.
 */
function genTrackId (): IdType {
  return uid()
}

/**
 * Generate new shape id. It should not be called outside this file.
 */
export function genShapeId (): IdType {
  return uid()
}

/**
 * Generate new item id. It should not be called outside this file.
 */
function genItemId (): IdType {
  return uid()
}

/**
 * Generate an integer representation with low collision for different ids
 * This is currently a sum of the char codes
 * @param {IdType} id
 */
export function id2int (s: IdType): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    // tslint:disable-next-line: no-bitwise
    h = h + s.charCodeAt(i) | 0
    // h = Math.imul(31, h) + s.charCodeAt(i) | 0
  }
  return h
}
