// @flow
/* eslint-disable new-cap */

import {Map, List, Record} from 'immutable';

import type {
  LabelType,
  ItemType,
  RectType,
  SatType,
  SatConfigType, SatCurrentType,
} from './types';

export const newLabel: LabelType = Record({
  id: -1,
  item: -1, // id
  categoryPath: '',
  attributes: Map(),
  parent: -1, // id
  children: List(), // ids
  numChildren: 0,
  valid: true,
  shapes: List(),
  selectedShape: -1,
  state: -1,
});

export const newRect: RectType = Record({
  id: -1,
  x1: -1,
  y1: -1,
  x2: -1,
  y2: -1,
});

export const newItem: ItemType = Record({
  id: -1,
  index: 0,
  url: '',
  active: false,
  labels: List(), // list of label ids
  attributes: Map(),
});

export const SatConfigS: SatConfigType = Record({
  assignmentId: '', // id
  projectName: '',
  itemType: '',
  labelType: '',
  taskSize: 0,
  handlerUrl: '',
  pageTitle: '',
  instructionPage: '', // instruction url
  demoMode: false,
  categories: List(),
  attributes: Map(),
  taskId: '',
  workerId: '',
  startTime: 0,
});

export const SatCurrentS: SatCurrentType = Record({
  item: -1,
  label: -1,
  maxObjectId: -1,
});

export const newSat: SatType = Record({
  config: SatConfigS(),
  current: SatCurrentS(),
  items: List(), // Map from item index to item, ordered
  labels: Map(), // Map from label id to label
  tracks: Map(),
  shapes: Map(), // Map from shapeId to shape
  actions: List(),
});
