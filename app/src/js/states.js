// @flow

import type {
  LabelType, ItemType,
  RectType, SatType,
  SatConfigType, SatCurrentType,
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
    numChildren: 0,
    valid: true,
    shapes: [],
    selectedShape: -1,
    state: -1,
    ...params,
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
    x1: -1,
    y1: -1,
    x2: -1,
    y2: -1,
    ...params,
  };
}

/**
 * Initialize an item state
 * @param {{}} params
 * @return {ItemType}
 */
export function makeItem(params: {} = {}): ItemType {
  return {
    id: -1,
    index: 0,
    url: '',
    active: false,
    loaded: false,
    labels: [], // list of label ids
    ...params,
  };
}

/**
 * Make Sat configuration state
 * @param {{}} params
 * @return {SatConfigType}
 */
export function makeSatConfig(params: {} = {}): SatConfigType {
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
    ...params,
  };
}

/**
 * Initialize a Sat current state
 * @param {{}} params
 * @return {SatCurrentType}
 */
export function makeSatCurrent(params: {} = {}): SatCurrentType {
  return {
    item: -1,
    label: -1,
    maxObjectId: -1,
    ...params,
  };
}

/**
 * Initialize a Sat state
 * @param {{}} params
 * @return {SatType}
 */
export function makeSat(params: {} = {}): SatType {
  return {
    config: makeSatConfig(),
    current: makeSatCurrent(),
    items: [], // Map from item index to item, ordered
    labels: {}, // Map from label id to label
    tracks: {},
    shapes: {}, // Map from shapeId to shape
    actions: [],
    ...params,
  };
}
