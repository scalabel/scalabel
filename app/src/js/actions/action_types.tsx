// SAT specific actions
// separate into activate and deactivate?
// no need if the two are always called together
import {
  ItemType,
  ViewerConfigType,
  LabelType,
  Vector3Type
} from '../functional/types';

export const INIT_SESSION = 'INIT_SESSION';
export const NEW_ITEM = 'NEW_ITEM'; // no delete item
export const GO_TO_ITEM = 'GO_TO_ITEM';
export const LOAD_ITEM = 'LOAD_ITEM';
export const UPDATE_ALL = 'UPDATE_ALL';

export const IMAGE_ZOOM = 'IMAGE_ZOOM';

// Item Level
export const NEW_LABEL = 'NEW_LABEL';
export const DELETE_LABEL = 'DELETE_LABEL';
// Image specific actions
export const TAG_IMAGE = 'TAG_IMAGE';

// Label Level
export const CHANGE_ATTRIBUTE = 'CHANGE_ATTRIBUTE';
export const CHANGE_CATEGORY = 'CHANGE_CATEGORY';

// View Level
export const TOGGLE_ASSISTANT_VIEW = 'TOGGLE_ASSISTANT_VIEW';

// Box2D specific
export const NEW_IMAGE_BOX2D_LABEL = 'NEW_IMAGE_BOX2D_LABEL';
export const CHANGE_RECT = 'CHANGE_RECT';

// Point Cloud Specific
export const MOVE_CAMERA = 'MOVE_CAMERA';
export const MOVE_CAMERA_AND_TARGET = 'MOVE_CAMERA_AND_TARGET';

export interface InitSessionAction {
  type: typeof INIT_SESSION;
}

export interface NewItemAction {
  type: typeof NEW_ITEM;
  createItem: (itemId: number, url: string) => ItemType;
  url: string;
}

export interface GoToItemAction {
  type: typeof GO_TO_ITEM;
  index: number;
}

export interface LoadItemAction {
  type: typeof LOAD_ITEM;
  index: number;
  config: ViewerConfigType;
}

export interface UpdateAllAction {
  type: typeof UPDATE_ALL;
}

export interface ImageZoomAction {
  type: typeof IMAGE_ZOOM;
  ratio: number;
}

export interface NewLabelAction {
  type: typeof NEW_LABEL;
  itemId: number;
  createLabel: (labelId: number, itemId: number, optionalAttributes: any) =>
    LabelType;
  optionalAttributes: any;
}

export interface DeleteLabelAction {
  type: typeof DELETE_LABEL;
  itemId: number;
  labelId: number;
}

export interface TagImageAction {
  type: typeof TAG_IMAGE;
  itemId: number;
  attributeIndex: number;
  selectedIndex: number[];
}

export interface ChangeAttributeAction {
  type: typeof CHANGE_ATTRIBUTE;
  labelId: number;
  attributeOptions: any;
}

export interface ChangeCategoryAction {
  type: typeof CHANGE_CATEGORY;
  labelId: number;
  categoryOptions: any;
}

export interface ToggleAssistantViewAction {
  type: typeof TOGGLE_ASSISTANT_VIEW;
}

export interface NewImageBox2dLabelAction {
  type: typeof NEW_IMAGE_BOX2D_LABEL;
  itemId: number;
  optionalAttributes: any;
}

export interface ChangeRectAction {
  type: typeof CHANGE_RECT;
  shapeId: number;
  targetBoxAttributes: any;
}

export interface MoveCameraAction {
  type: typeof MOVE_CAMERA;
  newPosition: Vector3Type;
}

export interface MoveCameraAndTargetAction {
  type: typeof MOVE_CAMERA_AND_TARGET;
  newPosition: Vector3Type;
  newTarget: Vector3Type;
}

export type ActionTypes =
  InitSessionAction | NewItemAction | GoToItemAction | LoadItemAction |
  UpdateAllAction | ImageZoomAction | NewLabelAction | DeleteLabelAction |
  TagImageAction | ChangeAttributeAction | ChangeCategoryAction |
  ToggleAssistantViewAction | NewImageBox2dLabelAction | ChangeRectAction |
  MoveCameraAction | MoveCameraAndTargetAction;
