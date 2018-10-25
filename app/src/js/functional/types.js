
export type LabelType = {
  id: number,
  item: number,
  category: Array<number>,
  attributes: {[string]: Array<number>},
  parent: number,
  children: Array<number>,
  valid: boolean,
  shapes: Array<number>,
  selectedShape: number,
  state: number
};

export type ShapeType = {
  id: number,
  label: number
};

export type RectType = {
  ...ShapeType,
  x1: number,
  y1: number,
  x2: number,
  y2: number
};

export type CubeType = {
  ...ShapeType,
  center: Array<number>,
  size: Array<number>,
  orientation: Array<number>
};

export type ImageViewerConfigType = {
  imageWidth: number,
  imageHeight: number,
  viewScale: number
};

export type PointCloudViewerConfigType = {
};

export type ViewerConfigType = Object;

export type ItemType = {
  id: number,
  index: number,
  url: string,
  active: boolean,
  loaded: boolean,
  labels: Array<number>, // list of label ids
  viewerConfig: ViewerConfigType
};

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt, put the props in config in favor
  of smaller SatProps.
 */
export type ConfigType = {
  assignmentId: string, // id
  projectName: string,
  itemType: string,
  labelType: string,
  taskSize: number,
  handlerUrl: string,
  pageTitle: string,
  instructionPage: string, // instruction url
  demoMode: boolean,
  bundleFile: string,
  categories: Array<string>,
  attributes: Array<Object>,
  taskId: string,
  workerId: string,
  startTime: number,
};

/*
  The current state of Sat.
 */
export type CurrentType = {
  item: number, // currently viewed item
  label: number, // currently selected label
  shape: number, // currently selected shape
  maxObjectId: number,
};

export type StateType = {
  config: ConfigType,
  current: CurrentType,
  items: Array<ItemType>,
  labels: {[number]: LabelType}, // Map from label id string to label
  tracks: {[number]: LabelType},
  shapes: {[number]: RectType | CubeType},
  actions: Array<any>,
};

export type LabelFunctionalType ={
  createLabel: (number, number, Object) => LabelType,
};

export type ItemFunctionalType ={
  createItem: (number, string) => ItemType,
  // setActive: (number, boolean) => ItemType,
};
