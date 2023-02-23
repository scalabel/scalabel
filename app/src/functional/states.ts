import { bool } from "aws-sdk/clients/signer"
import _ from "lodash"

import { uid } from "../common/uid"
import * as types from "../const/common"
import { ItemExport, LabelExport } from "../types/export"
import {
  Attribute,
  ColorSchemeType,
  ConfigType,
  ConnectionStatus,
  CubeType,
  ExtrinsicsType,
  HomographyViewerConfigType,
  IdType,
  Image3DViewerConfigType,
  ImageViewerConfigType,
  Info3DType,
  IntrinsicsType,
  INVALID_ID,
  ItemStatus,
  //OverlayStatus,
  ItemType,
  LabelType,
  LayoutType,
  ModeStatus,
  Node2DType,
  PaneType,
  PathPoint2DType,
  PathPointType,
  Plane3DType,
  PointCloudViewerConfigType,
  PolygonType,
  PolyPathPoint2DType,
  RectType,
  Select,
  SensorType,
  SessionType,
  ShapeType,
  SimplePathPoint2DType,
  SplitType,
  State,
  TaskStatus,
  TaskType,
  TrackType,
  UserType,
  ViewerConfigType
} from "../types/state"
import { taskIdToString } from "./id2string"

/**
 * Initialize a label state and deep copy the parameters.
 * Every label has an id when it is born.
 *
 * @param {Partial<LabelType>} params
 * @param {boolean} newId create new id even if it is in params
 * @returns {LabelType}
 */
export function makeLabel(
  params: Partial<LabelType> = {},
  newId: boolean = true
): LabelType {
  const label: LabelType = {
    id: genLabelId(),
    item: -1,
    sensors: [-1],
    type: types.LabelTypeName.EMPTY,
    category: [],
    attributes: {},
    parent: INVALID_ID, // Id
    children: [], // Ids
    shapes: [],
    track: INVALID_ID,
    order: 0,
    manual: true, // By default, manual is true
    changed: false, // If shape has changed, then interpolation will not apply
    checked: false,
    ..._.cloneDeep(params)
  }
  if (newId && params.id !== undefined) {
    label.id = genLabelId()
  }
  return label
}

/**
 * Initialize a track
 * Every track has an id when it is born
 *
 * @param {Partial<TrackType>} params
 * @param {boolean} newId create new id even if it is in params
 * Labels can not be filled without specifying id
 */
export function makeTrack(
  params: Partial<TrackType> = {},
  newId: boolean = true
): TrackType {
  const track: TrackType = {
    id: genTrackId(),
    type: types.LabelTypeName.EMPTY,
    labels: {},
    ...params
  }
  if (newId && params.id !== undefined) {
    track.id = genTrackId()
  }
  return track
}

/**
 * Make an empty shape
 * Every shape has an id when it is born
 *
 * @param {string} shapeType type name of the shape
 * @param params
 * @param {boolean} newId create new id even if it is in params
 * @param {bool} new
 */
export function makeShape(
  shapeType: string = "",
  params: Partial<ShapeType> = {},
  newId: bool = true
): ShapeType {
  const shape = {
    label: [],
    shapeType,
    id: genShapeId(),
    ..._.cloneDeep(params)
  }
  if (newId && params.id !== undefined) {
    shape.id = genShapeId()
  }
  return shape
}
/**
 * Initialize a rectangle shape
 *
 * @param {{}} params
 * @param newId
 * @returns {RectType}
 */
export function makeRect(
  params: Partial<RectType> = {},
  newId: bool = true
): RectType {
  return {
    x1: -1,
    y1: -1,
    x2: -1,
    y2: -1,
    ...makeShape(types.ShapeTypeName.RECT, params, newId)
  }
}

/**
 * Initialize a polygon
 *
 * @param {{}} params
 * @returns {PolygonType}
 */
export function makePolygon(params: Partial<PolygonType> = {}): PolygonType {
  return {
    points: [],
    ...makeShape(types.ShapeTypeName.POLYGON_2D),
    ...params
  }
}

/**
 * Initialize a polygon
 *
 * @param {{}} params
 * @returns {PolygonType}
 */
export function makePathPoint2D(
  params: Partial<PathPoint2DType> = {}
): PathPoint2DType {
  return {
    x: -1,
    y: -1,
    pointType: PathPointType.UNKNOWN,
    ...makeShape(types.ShapeTypeName.PATH_POINT_2D),
    ...params
  }
}

/**
 * Driver function to make a simple path point
 *
 * @param x
 * @param y
 * @param pointType
 */
export function makeSimplePathPoint2D(
  x: number,
  y: number,
  pointType: PathPointType
): SimplePathPoint2DType {
  return { x, y, pointType }
}

/**
 * Initialize a pathPoint shape
 *
 * @param params
 */
export function makePolyPathPoint(
  params: Partial<PolyPathPoint2DType> = {}
): PolyPathPoint2DType {
  return {
    x: -1,
    y: -1,
    pointType: "vertex",
    ...params
  }
}

/**
 * Initialize a 3d box shape
 *
 * @param {{}} params
 * @returns {CubeType}
 */
export function makeCube(params: Partial<CubeType> = {}): CubeType {
  return {
    center: { x: 0, y: 0, z: 0 },
    size: { x: 1, y: 1, z: 1 },
    orientation: { x: 0, y: 0, z: 0 },
    anchorIndex: 0,
    ...makeShape(types.ShapeTypeName.CUBE),
    ...params
  }
}

/**
 * Initialize a 3d box shape
 *
 * @param {{}} params
 * @returns {Plane3DType}
 */
export function makePlane(params: {} = {}): Plane3DType {
  return {
    center: { x: 0, y: 0, z: 0 },
    orientation: { x: 0, y: 0, z: 0 },
    ...makeShape(types.ShapeTypeName.GRID),
    ...params
  }
}

/**
 * Make a Node2D type
 *
 * @param params
 */
export function makeNode2d(params: Partial<Node2DType> = {}): Node2DType {
  return {
    name: "",
    hidden: false,
    x: -1,
    y: -1,
    ...makeShape(types.ShapeTypeName.NODE_2D),
    ...params
  }
}

/**
 * Create data source
 *
 * @param id
 * @param type
 * @param name
 * @param intrinsics
 * @param extrinsics
 */
export function makeSensor(
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
 *
 * @param pane
 * @param sensor
 * @returns {ImageViewerConfigType}
 */
export function makeImageViewerConfig(
  pane: number,
  sensor: number = -1
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
    hideLabels: false,
    hideTags: false
  }
}

/**
 * Make a new point cloud viewer config
 *
 * @param pane
 * @param sensor
 * @returns {PointCloudViewerConfigType}
 */
export function makePointCloudViewerConfig(
  pane: number,
  sensor: number = -1
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
    hideLabels: false,
    hideTags: false,
    cameraRotateDir: false,
    colorScheme: ColorSchemeType.HEIGHT,
    cameraTransformed: false
  }
}

/**
 * Make image 3d viewer config
 *
 * @param pane
 * @param sensor
 */
export function makeImage3DViewerConfig(
  pane: number,
  sensor: number = -1
): Image3DViewerConfigType {
  const imageConfig = makeImageViewerConfig(pane, sensor)
  return {
    ...imageConfig,
    target: { x: 0.0, y: 0.0, z: 1.0 },
    verticalAxis: { x: 0.0, y: -1.0, z: 0.0 },
    type: types.ViewerConfigTypeName.IMAGE_3D,
    pointCloudSensor: -2,
    pointCloudOverlay: false
  }
}

/**
 * Make homography viewer config
 *
 * @param pane
 * @param sensor
 * @param distance
 */
export function makeHomographyViewerConfig(
  pane: number,
  sensor: number = -1,
  distance: number = 10
): HomographyViewerConfigType {
  const imageConfig = makeImageViewerConfig(pane, sensor)
  return {
    ...imageConfig,
    target: { x: 0.0, y: 0.0, z: 1.0 },
    verticalAxis: { x: 0.0, y: -1.0, z: 0.0 },
    type: types.ViewerConfigTypeName.HOMOGRAPHY,
    pointCloudSensor: -2,
    distance,
    pointCloudOverlay: false
  }
}

/**
 * Create default viewer config for item type
 *
 * @param sensors
 * @param type
 * @param pane
 * @param sensor
 */
export function makeDefaultViewerConfig(
  type: types.ViewerConfigTypeName,
  pane: number = 0,
  sensor: number = -1
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
 * Make a default attribute
 *
 * @param params
 */
export function makeAttribute(params: Partial<Attribute> = {}): Attribute {
  return {
    type: types.AttributeToolType.SWITCH,
    name: "",
    values: [],
    tag: "",
    tagPrefix: "",
    tagSuffixes: [],
    buttonColors: [],
    ...params
  }
}
/**
 * Initialize an item state
 *
 * @param {{}} params
 * @param {boolean} keepId Use the input param id instead of generating a new ID
 * @returns {ItemType}
 */
export function makeItem(
  params: Partial<ItemType> = {},
  keepId: boolean = false
): ItemType {
  const item: ItemType = {
    id: genItemId(),
    index: 0,
    videoName: "",
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
 *
 * @param {{}} params
 * @returns {ItemExport}
 */
export function makeItemExport(params: Partial<ItemExport> = {}): ItemExport {
  return {
    name: "",
    url: "",
    videoName: "",
    sensor: -1,
    attributes: {},
    timestamp: -1,
    labels: [],
    ...params
  }
}

/**
 * Initialize an exportable label
 *
 * @param {{}} params
 * @returns {ItemExport}
 */
export function makeLabelExport(
  params: Partial<LabelExport> = {}
): LabelExport {
  return {
    id: genLabelId(),
    category: "",
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
 *
 * @param {{}} params
 * @returns {ConfigType}
 */
export function makeTaskConfig(params: Partial<ConfigType> = {}): ConfigType {
  return {
    projectName: "",
    itemType: "",
    labelTypes: [],
    label2DTemplates: {},
    policyTypes: [],
    taskSize: 0,
    keyInterval: 1,
    tracking: false,
    handlerUrl: "",
    pageTitle: "",
    instructionPage: "", // Instruction url
    bundleFile: "",
    categories: [],
    treeCategories: [],
    attributes: [],
    taskId: "",
    demoMode: false,
    autosave: false,
    bots: false,
    ...params
  }
}

/**
 * Make pane node
 *
 * @param viewerId
 * @param size
 * @param paneId
 * @param parent
 * @param primarySize
 * @param split
 * @param minSize
 * @param maxSize
 * @param primary
 * @param minPrimarySize
 * @param maxPrimarySize
 * @param child1
 * @param child2
 */
export function makePane(
  viewerId: number = -1,
  paneId: number = -1,
  parent: number = -1,
  primarySize?: number,
  split?: SplitType,
  primary?: "first" | "second",
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
 *
 * @param {{}} params
 * @returns {LayoutType}
 */
export function makeLayout(params: {} = {}): LayoutType {
  return {
    toolbarWidth: 200,
    maxViewerConfigId: 0,
    maxPaneId: 0,
    rootPane: 0,
    panes: { 0: makePane(0, 0) },
    ...params
  }
}

/**
 * Initialize a user selection sate
 *
 * @param {{}} params
 * @returns {Selection}
 */
function makeSelect(params: Partial<Select> = {}): Select {
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
 *
 * @param {{}} params
 * @returns {UserType}
 */
function makeUser(params: Partial<UserType> = {}): UserType {
  return {
    id: INVALID_ID,
    select: makeSelect(),
    layout: makeLayout(),
    viewerConfigs: [],
    ...params
  }
}

/**
 * Initialize a item status sate
 *
 * @param {{}} params
 * @returns {ItemStatus}
 */
export function makeItemStatus(params: Partial<ItemStatus> = {}): ItemStatus {
  return {
    sensorDataLoaded: {},
    ...params
  }
}




/**
 * Initialize a item status sate
 *
 * @param {{}} params
 * @returns {ItemStatus}
 */
export function makeInfo3D(params: Partial<Info3DType> = {}): Info3DType {
  return {
    isBoxSpan: false,
    boxSpan: null,
    showGroundPlane: false,
    ...params
  }
}

/**
 * Initialize a session state
 *
 * @param {{}} params
 * @returns {Session}
 */
function makeSession(params: Partial<SessionType> = {}): SessionType {
  return {
    id: INVALID_ID,
    startTime: 0,
    itemStatuses: [],
    trackLinking: false,
    polygon2DBoundaryClone: undefined,
    status: ConnectionStatus.UNSAVED,
    mode: ModeStatus.ANNOTATING,
    numUpdates: 0,
    alerts: [],
    info3D: makeInfo3D(),
    overlayStatus: [],
    overlayTransparency: 1,
    radarStatus: [],
    ...params
  }
}

/**
 * Initialize a task status state
 *
 * @param {{}} params
 * @returns {TaskStatus}
 */
function makeTaskStatus(params: Partial<TaskStatus> = {}): TaskStatus {
  return {
    maxOrder: -1,
    ...params
  }
}

/**
 * Initialize a task sate
 *
 * @param {{}} params
 * @returns {TaskType}
 */
export function makeTask(params: Partial<TaskType> = {}): TaskType {
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
 *
 * @param {{}} params
 * @returns {State}
 */
export function makeState(params: Partial<State> = {}): State {
  return {
    task: makeTask(params.task),
    user: makeUser(params.user),
    session: makeSession(params.session)
  }
}

/**
 * Check whether the input ID is valid or not default
 *
 * @param {IdType} id
 */
export function isValidId(id: IdType): bool {
  return id !== INVALID_ID && id !== "-1"
}

/**
 * Generate new label id. It should not be called outside this file.
 */
function genLabelId(): IdType {
  return uid()
}

/**
 * Generate new track id. It should not be called outside this file.
 */
function genTrackId(): IdType {
  return uid()
}

/**
 * Generate new shape id. It should not be called outside this file.
 */
function genShapeId(): IdType {
  return uid()
}

/**
 * Generate new item id. It should not be called outside this file.
 */
function genItemId(): IdType {
  return uid()
}

/**
 * Generate an integer representation with low collision for different ids
 * This is currently a sum of the char codes
 *
 * @param {IdType} id
 * @param s
 */
export function id2int(s: IdType): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    // eslint-disable-next-line no-bitwise
    h = (h + s.charCodeAt(i)) | 0
    // H = Math.imul(31, h) + s.charCodeAt(i) | 0
  }
  return h
}
