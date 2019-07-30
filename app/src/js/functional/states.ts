import uuid from 'uuid/v4'
import * as labels from '../common/label_types'
import {
  ConfigType, CubeType,
  CurrentType, ImageViewerConfigType, ItemType,
  LabelType, LayoutType, PointCloudViewerConfigType, RectType,
  State
} from './types'

/**
 * Initialize a label state
 * @param {{}} params
 * @return {LabelType}
 */
export function makeLabel (params: Partial<LabelType> = {}): LabelType {
  return {
    id: -1,
    item: -1,
    type: labels.EMPTY,
    category: [],
    attributes: {},
    parent: -1, // id
    children: [], // ids
    shapes: [],
    selectedShape: -1,
    state: -1,
    order: 0,
    ...params
  }
}

/**
 * Initialize a rectangle shape
 * @param {{}} params
 * @return {RectType}
 */
export function makeRect (params: Partial<RectType> = {}): RectType {
  return {
    id: -1,
    label: -1,
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
export function makeCube (params: {} = {}): CubeType {
  return {
    id: -1,
    label: -1,
    center: { x: 0, y: 0, z: 0 },
    size: { x: 1, y: 1, z: 1 },
    orientation: { x: 0, y: 0, z: 0 },
    ...params
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
export function makeItem (params: {} = {}): ItemType {
  return {
    id: -1,
    index: 0,
    url: '',
    active: false,
    loaded: false,
    labels: {},
    shapes: {},
    viewerConfig: null,
    ...params
  }
}

/**
 * Make Sat configuration state
 * @param {{}} params
 * @return {ConfigType}
 */
export function makeSatConfig (params: Partial<ConfigType> = {}): ConfigType {
  return {
    sessionId: uuid(),
    assignmentId: '', // id
    projectName: '',
    itemType: '',
    labelTypes: [],
    taskSize: 0,
    handlerUrl: '',
    pageTitle: '',
    instructionPage: '', // instruction url
    demoMode: false,
    bundleFile: '',
    categories: [],
    attributes: [],
    taskId: '',
    workerId: '',
    startTime: 0,
    ...params
  }
}

/**
 * Initialize a Sat current state
 * @param {{}} params
 * @return {CurrentType}
 */
export function makeSatCurrent (
    params: Partial<CurrentType> = {}): CurrentType {
  return {
    item: -1,
    label: -1,
    shape: -1,
    category: 0,
    labelType:  0,
    maxLabelId: -1,
    maxShapeId: -1,
    maxOrder: -1,
    ...params
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
    assistantView: false,
    assistantViewRatio: 0.3,
    ...params
  }
}

/**
 * Initialize a Sat state
 * @param {{}} params
 * @return {State}
 */
export function makeState (params: Partial<State> = {}): State {
  params.config = makeSatConfig(params.config)
  params.current = makeSatCurrent(params.current)
  const state = {
    config: makeSatConfig(),
    current: makeSatCurrent(),
    items: [], // Map from item index to item, ordered
    tracks: {},
    layout: makeLayout(),
    ...params
  }
  return state
}
