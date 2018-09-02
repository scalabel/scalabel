/* @flow */

import type {Map, List, RecordFactory, RecordOf} from 'immutable';

type LabelProps = {
  id: number,
  item: number,
  categoryPath: string,
  attributes: Map<string, string | number>,
  parent: number,
  children: List<number>,
  valid: boolean,
  shapes: List<number>,
  selectedShape: number,
  state: number
};

export type LabelType = RecordFactory<LabelProps>;

type RectProps = {
  id: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
};

export type RectType = RecordFactory<RectProps>;

type CubeProps = {
  id: number,
  center: List<number>,
  size: List<number>,
  orientation: List<number>
};

export type CubeType = RecordFactory<CubeProps>;

type ItemProps = {
  id: number,
  index: number,
  url: string,
  active: boolean,
  labels: List<number>, // list of label ids
  attributes: Map<string, string>
};

export type ItemType = RecordFactory<ItemProps>;

export type SatConfigProps = {
  assignmentId: string, // id
  projectName: string,
  itemType: string,
  labelType: string,
  taskSize: number,
  handlerUrl: string,
  pageTitle: string,
  instructionPage: string, // instruction url
  demoMode: boolean,
  categories: List<string>,
  attributes: Map<string, string | number>,
  taskId: string,
  workerId: string
};

export type SatConfigType = RecordFactory<SatConfigProps>;

type SatProps = {
  config: RecordOf<SatConfigProps>,
  items: List<ItemType>,
  labels: Map<number, LabelType>, // Map from label id to label
  tracks: Map<number, LabelType>,
  shapes: Map<number, RectType | CubeType>,
  actions: List<any>,
  currentItem: number, // id of current item
  maxObjectId: number,
  startTime: number,
};

export type SatType = RecordFactory<SatProps>;
