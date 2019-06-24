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
  /** Type of the action */
  type: typeof INIT_SESSION;
}

export interface NewItemAction {
  /** Type of the action */
  type: typeof NEW_ITEM;
  /** Function of createItem */
  createItem: (itemId: number, url: string) => ItemType;
  /** The url */
  url: string;
}

export interface GoToItemAction {
  /** Type of the action */
  type: typeof GO_TO_ITEM;
  /** Index of the item to go to */
  index: number;
}

export interface LoadItemAction {
  /** Type of the action */
  type: typeof LOAD_ITEM;
  /** Index of the item to load */
  index: number;
  /** Configurations */
  config: ViewerConfigType;
}

export interface UpdateAllAction {
  /** Type of the action */
  type: typeof UPDATE_ALL;
}

export interface ImageZoomAction {
  /** Type of the action */
  type: typeof IMAGE_ZOOM;
  /** Zoom ratio */
  ratio: number;
  /** View Offset X */
  viewOffsetX: number;
  /** View Offset Y */
  viewOffsetY: number;
}

export interface NewLabelAction {
  /** Type of the action */
  type: typeof NEW_LABEL;
  /** Item ID of the new label */
  itemId: number;
  /** Create label function */
  createLabel: (labelId: number, itemId: number, optionalAttributes: any) =>
    LabelType;
  /** Optional attributes */
  optionalAttributes: any;
}

export interface DeleteLabelAction {
  /** Type of the action */
  type: typeof DELETE_LABEL;
  /** ID of the corresponding item */
  itemId: number;
  /** ID of label to be deleted */
  labelId: number;
}

export interface TagImageAction {
  /** Type of the action */
  type: typeof TAG_IMAGE;
  /** ID of the corresponding item */
  itemId: number;
  /** Index of the attribute */
  attributeIndex: number;
  /** Index of the selected attribute */
  selectedIndex: number[];
}

export interface ChangeAttributeAction {
  /** Type of the action */
  type: typeof CHANGE_ATTRIBUTE;
  /** ID of the label */
  labelId: number;
  /** Attribute options */
  attributeOptions: any;
}

export interface ChangeCategoryAction {
  /** Type of the action */
  type: typeof CHANGE_CATEGORY;
  /** ID of the label */
  labelId: number;
  /** Category options */
  categoryOptions: any;
}

export interface ToggleAssistantViewAction {
  /** Type of the action */
  type: typeof TOGGLE_ASSISTANT_VIEW;
}

export interface NewImageBox2dLabelAction {
  /** Type of the action */
  type: typeof NEW_IMAGE_BOX2D_LABEL;
  /** Item of the corresponding item */
  itemId: number;
  /** Optional attributes */
  optionalAttributes: any;
}

export interface ChangeRectAction {
  /** Type of the action */
  type: typeof CHANGE_RECT;
  /** ID of the shape */
  shapeId: number;
  /** Target box attributes */
  targetBoxAttributes: any;
}

export interface MoveCameraAction {
  /** Type of the action */
  type: typeof MOVE_CAMERA;
  /** New position */
  newPosition: Vector3Type;
}

export interface MoveCameraAndTargetAction {
  /** Type of the action */
  type: typeof MOVE_CAMERA_AND_TARGET;
  /** New position */
  newPosition: Vector3Type;
  /** New target */
  newTarget: Vector3Type;
}

export type ActionTypes =
  InitSessionAction | NewItemAction | GoToItemAction | LoadItemAction |
  UpdateAllAction | ImageZoomAction | NewLabelAction | DeleteLabelAction |
  TagImageAction | ChangeAttributeAction | ChangeCategoryAction |
  ToggleAssistantViewAction | NewImageBox2dLabelAction | ChangeRectAction |
  MoveCameraAction | MoveCameraAndTargetAction;
