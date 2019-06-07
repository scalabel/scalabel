export interface LabelType {
  id: number;
  item: number;
  category: number[];
  attributes: {[key: number]: number[]};
  parent: number;
  children: number[];
  valid: boolean;
  shapes: number[];
  selectedShape: number;
  state: number;
}

export interface ShapeType {
  id: number;
  label: number;
}

export interface RectType extends ShapeType {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Vector3Type {
  x: number;
  y: number;
  z: number;
}

export interface CubeType extends ShapeType {
  center: Vector3Type;
  size: Vector3Type;
  orientation: Vector3Type;
}

export interface ImageViewerConfigType {
  imageWidth: number;
  imageHeight: number;
  viewScale: number;
}

export interface PointCloudViewerConfigType {
  position: Vector3Type;
  target: Vector3Type;
  verticalAxis: Vector3Type;
}

export type ViewerConfigType =
    ImageViewerConfigType | PointCloudViewerConfigType;

export interface ItemType {
  id: number;
  index: number;
  url: string;
  active: boolean;
  loaded: boolean;
  labels: number[]; // list of label ids
  viewerConfig: ViewerConfigType;
}

/*
  Those properties are not changed during the lifetime of a session.
  It also make SatProps smaller. When in doubt; put the props in config in favor
  of smaller SatProps.
 */
export interface ConfigType {
  assignmentId: string; // id
  projectName: string;
  itemType: string;
  labelType: string;
  taskSize: number;
  handlerUrl: string;
  pageTitle: string;
  instructionPage: string; // instruction url
  demoMode: boolean;
  bundleFile: string;
  categories: string[];
  attributes: object[];
  taskId: string;
  workerId: string;
  startTime: number;
}

export interface LayoutType {
  toolbarWidth: number;
  assistantView: boolean;
  assistantViewRatio: number;
}

/*
  The current state of Sat.
 */
export interface CurrentType {
  item: number; // currently viewed item
  label: number; // currently selected label
  shape: number; // currently selected shape
  maxObjectId: number;
}

export interface StateType {
  config: ConfigType;
  current: CurrentType;
  items: ItemType[];
  labels: {[key: number]: LabelType}; // Map from label id string to label
  tracks: {[key: number]: LabelType};
  shapes: {[key: number]: any};
  actions: any[];
  layout: LayoutType;
}

export type LabelFunctionalType =
    (id: number, itemId: number, attributes: object) => LabelType;

export type ItemFunctionalType = (id: number, url: string) => ItemType;
