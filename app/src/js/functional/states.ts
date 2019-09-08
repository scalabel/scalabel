import _ from 'lodash'
import { LabelTypes } from '../common/types'
import {
  ConfigType, CubeType,
  ImageViewerConfigType, IndexedShapeType,
  ItemStatus, ItemType, LabelType, LayoutType,
  PointCloudViewerConfigType,
  RectType,
  Select,
  SessionType,
  ShapeType,
  State,
  TaskStatus,
  TaskType,
  UserType
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
    type: LabelTypes.EMPTY,
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
 * Make a new viewer config
 * @return {ImageViewerConfigType}
 */
export function makeImageViewerConfig (): ImageViewerConfigType {
  return {
    imageWidth: 0,
    imageHeight: 0,
    viewScale: 1.0,
    viewOffsetX: -1,
    viewOffsetY: -1
  }
}

/**
 * Make a new point cloud viewer config
 * @return {PointCloudViewerConfigType}
 */
export function makePointCloudViewerConfig (): PointCloudViewerConfigType {
  return {
    position: { x: 0.0, y: 10.0, z: 0.0 },
    target: { x: 0.0, y: 0.0, z: 0.0 },
    verticalAxis: { x: 0.0, y: 0.0, z: 1.0 }
  }
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
    url: '',
    labels: {},
    shapes: {},
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
    taskSize: 0,
    handlerUrl: '',
    pageTitle: '',
    instructionPage: '', // instruction url
    bundleFile: '',
    categories: [],
    attributes: [],
    taskId: '',
    submitTime: 0,
    ...params
  }
}

/**
 * Initialize a Layout state
 * @param {{}} params
 * @return {LayoutType}
 */
function makeLayout (params: {} = {}): LayoutType {
  return {
    toolbarWidth: 200,
    assistantView: false,
    assistantViewRatio: 0.3,
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
    item: -1,
    label: -1,
    shape: -1,
    category: 0,
    labelType: 0,
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
    imageViewerConfig: makeImageViewerConfig(),
    pointCloudViewerConfig: makePointCloudViewerConfig(),
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
    loaded: false,
    ...params
  }
}

/**
 * Initialize a session sate
 * @param {{}} params
 * @return {Session}
 */
function makeSession (params: Partial<SessionType>= {}): SessionType {
  return {
    id: '',
    demoMode: false,
    startTime: 0,
    items: [],
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
    ...params
  }
}

/**
 * Initialize a task sate
 * @param {{}} params
 * @return {TaskType}
 */
function makeTask (params: Partial<TaskType> = {}): TaskType {
  return {
    config: makeTaskConfig(),
    status: makeTaskStatus(),
    items: [],
    tracks: {},
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
