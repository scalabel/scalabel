// @flow

import type {
  LabelType, ItemType,
  RectType, SatType,
  SatConfigType, SatCurrentType,
} from './types';

export const makeLabel = function(params: {} = {}): LabelType {
  return {
    id: -1,
    item: -1, // id
    categoryPath: '',
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
};

export const makeRect = function(params: {} = {}): RectType {
  return {
    id: -1,
    x1: -1,
    y1: -1,
    x2: -1,
    y2: -1,
    ...params,
  };
};

export const makeItem = function(params: {} = {}): ItemType {
  return {
    id: -1,
    index: 0,
    url: '',
    active: false,
    labels: [], // list of label ids
    attributes: {},
    ...params,
  };
};

export const makeSatConfig = function(params: {} = {}): SatConfigType {
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
    categories: [],
    attributes: {},
    taskId: '',
    workerId: '',
    startTime: 0,
    ...params,
  };
};

const makeSatCurrent = function(params: {} = {}): SatCurrentType {
  return {
    item: -1,
    label: -1,
    maxObjectId: -1,
    ...params,
  };
};

export const makeSat = function(config: {} = {}): SatType {
  return {
    config: makeSatConfig(config),
    current: makeSatCurrent(),
    items: [], // Map from item index to item, ordered
    labels: {}, // Map from label id to label
    tracks: {},
    shapes: {}, // Map from shapeId to shape
    actions: [],
  };
};
