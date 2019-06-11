// @flow

import {
  LabelType, ItemType,
  RectType, CubeType, StateType,
  ConfigType, CurrentType, ImageViewerConfigType, PointCloudViewerConfigType,
  LayoutType,
} from './types';

/**
 * Initialize a label state
 * @param {{}} params
 * @return {LabelType}
 */
export function makeLabel(params: {} = {}): LabelType {
  return {
    id: -1,
    item: -1, // id
    category: [],
    attributes: {},
    parent: -1, // id
    children: [], // ids
    valid: true,
    shapes: [],
    selectedShape: -1,
    state: -1,
    ...params
  };
}

/**
 * Initialize a rectangle shape
 * @param {{}} params
 * @return {RectType}
 */
export function makeRect(params: {} = {}): RectType {
  return {
    id: -1,
    label: -1,
    x: -1,
    y: -1,
    w: -1,
    h: -1,
    ...params
  };
}

/**
 * Initialize a 3d box shape
 * @param {{}} params
 * @return {CubeType}
 */
export function makeCube(params: {} = {}): CubeType {
  return {
    id: -1,
    label: -1,
    center: {x: 0, y: 0, z: 0},
    size: {x: 1, y: 1, z: 1},
    orientation: {x: 0, y: 0, z: 0},
    ...params
  };
}

/**
 * Make a new viewer config
 * @return {ImageViewerConfigType}
 */
export function makeImageViewerConfig(): ImageViewerConfigType {
  return {
    imageWidth: 0,
    imageHeight: 0,
    viewScale: 1.0
  };
}

/**
 * Make a new point cloud viewer config
 * @return {PointCloudViewerConfigType}
 */
export function makePointCloudViewerConfig(): PointCloudViewerConfigType {
  return {
    position: {x: 0.0, y: 10.0, z: 0.0},
    target: {x: 0.0, y: 0.0, z: 0.0},
    verticalAxis: {x: 0.0, y: 0.0, z: 1.0}
  };
}

/**
 * Initialize an item state
 * @param {{}} params
 * @return {ItemType}
 */
export function makeItem(params: any = {}): ItemType {
  return {
    id: -1,
    index: 0,
    url: '',
    active: false,
    loaded: false,
    labels: [], // list of label ids
    viewerConfig: null,
    ...params
  };
}

/**
 * Make Sat configuration state
 * @param {{}} params
 * @return {ConfigType}
 */
export function makeSatConfig(params: {} = {}): ConfigType {
  return {
    assignmentId: '', // id
    projectName: '',
    itemType: '',
    labelType: '',
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
  };
}

/**
 * Initialize a Sat current state
 * @param {{}} params
 * @return {CurrentType}
 */
export function makeSatCurrent(params: {} = {}): CurrentType {
  return {
    item: -1,
    label: -1,
    shape: -1,
    maxObjectId: -1,
    ...params
  };
}

/**
 * Initialize a Layout state
 * @param {{}} params
 * @return {LayoutType}
 */
export function makeLayout(params: {} = {}): LayoutType {
  return {
    toolbarWidth: 200,
    assistantView: false,
    assistantViewRatio: 0.3,
    ...params
  };
}

/**
 * Initialize a Sat state
 * @param {{}} params
 * @return {StateType}
 */
export function makeState(params: {} = {}): StateType {
  return {
    config: makeSatConfig(),
    current: makeSatCurrent(),
    items: [], // Map from item index to item, ordered
    labels: {}, // Map from label id to label
    tracks: {},
    shapes: {}, // Map from shapeId to shape
    actions: [],
    layout: makeLayout(),
    ...params
  };
}
