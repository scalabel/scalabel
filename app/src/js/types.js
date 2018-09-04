/* @flow */

export type LabelProps = {
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

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt, put the props in config in favor
  of smaller SatProps.
 */
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
  workerId: string,
  startTime: number,
};

export type SatConfigType = RecordFactory<SatConfigProps>;

/*
  The current state of Sat.
 */
type SatCurrentProps = {
  item: number,
  label: number,
  maxObjectId: number,
};

export type SatCurrentType = RecordFactory<SatCurrentProps>;

type SatProps = {
  config: RecordOf<SatConfigProps>,
  current: RecordOf<SatCurrentProps>,
  items: List<ItemType>,
  labels: Map<number, LabelType>, // Map from label id to label
  tracks: Map<number, LabelType>,
  shapes: Map<number, RectType | CubeType>,
  actions: List<any>,
};

export type SatType = RecordFactory<SatProps>;
