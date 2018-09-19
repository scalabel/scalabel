/* @flow */

export type LabelType = {
  id: number,
  item: number,
  categoryPath: string,
  attributes: {[string]: string | number},
  parent: number,
  children: Array<number>,
  valid: boolean,
  shapes: Array<number>,
  selectedShape: number,
  state: number
};

export type RectType = {
  id: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
};

export type CubeType = {
  id: number,
  center: Array<number>,
  size: Array<number>,
  orientation: Array<number>
};

export type ItemType = {
  id: number,
  index: number,
  url: string,
  active: boolean,
  loaded: boolean,
  labels: Array<number>, // list of label ids
  attributes: {[string]: string}
};

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt, put the props in config in favor
  of smaller SatProps.
 */
export type SatConfigType = {
  assignmentId: string, // id
  projectName: string,
  itemType: string,
  labelType: string,
  taskSize: number,
  handlerUrl: string,
  pageTitle: string,
  instructionPage: string, // instruction url
  demoMode: boolean,
  categories: Array<string>,
  attributes: { [string]: string | number },
  taskId: string,
  workerId: string,
  startTime: number,
};

/*
  The current state of Sat.
 */
export type SatCurrentType = {
  item: number,
  label: number,
  maxObjectId: number,
};

export type SatType = {
  config: SatConfigType,
  current: SatCurrentType,
  items: Array<ItemType>,
  labels: {[number]: LabelType}, // Map from label id to label
  tracks: {[number]: LabelType},
  shapes: {[number]: RectType | CubeType},
  actions: Array<any>,
};

export type LabelFunctionalType ={
  createLabel: (number) => LabelType,
};
