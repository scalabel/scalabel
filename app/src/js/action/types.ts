// SAT specific actions
// separate into activate and deactivate?
// no need if the two are always called together
import {
  ItemType,
  LabelType,
  ShapeType,
  Vector3Type, ViewerConfigType
} from '../functional/types'

export const INIT_SESSION = 'INIT_SESSION'
export const NEW_ITEM = 'NEW_ITEM' // no delete item
export const GO_TO_ITEM = 'GO_TO_ITEM'
export const LOAD_ITEM = 'LOAD_ITEM'
export const UPDATE_ALL = 'UPDATE_ALL'

export const IMAGE_ZOOM = 'IMAGE_ZOOM'

// Item Level
export const ADD_LABEL = 'ADD_LABEL'
export const CHANGE_LABEL_SHAPE = 'CHANGE_LABEL_SHAPE'
export const CHANGE_LABEL_PROPS = 'CHANGE_LABEL_PROPS'
export const DELETE_LABEL = 'DELETE_LABEL'
// Image specific actions
export const TAG_IMAGE = 'TAG_IMAGE'

// Label Level
export const CHANGE_ATTRIBUTE = 'CHANGE_ATTRIBUTE'
export const CHANGE_CATEGORY = 'CHANGE_CATEGORY'

// View Level
export const TOGGLE_ASSISTANT_VIEW = 'TOGGLE_ASSISTANT_VIEW'

// Box2D specific
export const NEW_IMAGE_BOX2D_LABEL = 'NEW_IMAGE_BOX2D_LABEL'
export const CHANGE_RECT = 'CHANGE_RECT'

// Point Cloud Specific
export const MOVE_CAMERA = 'MOVE_CAMERA'
export const MOVE_CAMERA_AND_TARGET = 'MOVE_CAMERA_AND_TARGET'

export interface InitSessionAction {
  /** Type of the action */
  type: typeof INIT_SESSION
}

export interface NewItemAction {
  /** Type of the action */
  type: typeof NEW_ITEM
  /** Function of createItem */
  createItem: (itemId: number, url: string) => ItemType
  /** The url */
  url: string
}

export interface GoToItemAction {
  /** Type of the action */
  type: typeof GO_TO_ITEM
  /** Index of the item to go to */
  index: number
}

export interface LoadItemAction {
  /** Type of the action */
  type: typeof LOAD_ITEM
  /** Index of the item to load */
  index: number
  /** Configurations */
  config: ViewerConfigType
}

export interface UpdateAllAction {
  /** Type of the action */
  type: typeof UPDATE_ALL
}

export interface ImageZoomAction {
  /** Type of the action */
  type: typeof IMAGE_ZOOM
  /** Zoom ratio */
  ratio: number
  /** View Offset X */
  viewOffsetX: number
  /** View Offset Y */
  viewOffsetY: number
}

export interface AddLabelAction {
  /** Type of the action */
  type: typeof ADD_LABEL
  /** label to add */
  label: LabelType
  /** Shapes of the label */
  shapes: ShapeType[]
}

export interface ChangeLabelShapeAction {
  /** Type of the action */
  type: typeof CHANGE_LABEL_SHAPE
  /** Shape ID */
  shapeId: number
  /** properties to update for the shape */
  props: object
}

export interface ChangeLabelPropsAction {
  /** Type of the action */
  type: typeof CHANGE_LABEL_PROPS
  /** Label ID */
  labelId: number
  /** properties to update for the shape */
  props: object
}

export interface DeleteLabelAction {
  /** Type of the action */
  type: typeof DELETE_LABEL
  /** ID of label to be deleted */
  labelId: number
}

export interface TagImageAction {
  /** Type of the action */
  type: typeof TAG_IMAGE
  /** ID of the corresponding item */
  itemId: number
  /** Index of the attribute */
  attributeIndex: number
  /** Index of the selected attribute */
  selectedIndex: number[]
}

export interface ChangeAttributeAction {
  /** Type of the action */
  type: typeof CHANGE_ATTRIBUTE
  /** ID of the label */
  labelId: number
  /** Attribute options */
  attributeOptions: object
}

export interface ChangeCategoryAction {
  /** Type of the action */
  type: typeof CHANGE_CATEGORY
  /** ID of the label */
  labelId: number
  /** Category options */
  categoryOptions: object
}

export interface ToggleAssistantViewAction {
  /** Type of the action */
  type: typeof TOGGLE_ASSISTANT_VIEW
}

export interface NewImageBox2dLabelAction {
  /** Type of the action */
  type: typeof NEW_IMAGE_BOX2D_LABEL
  /** Item of the corresponding item */
  itemId: number
  /** Optional attributes */
  optionalAttributes: object
}

export interface ChangeRectAction {
  /** Type of the action */
  type: typeof CHANGE_RECT
  /** ID of the shape */
  shapeId: number
  /** Target box attributes */
  targetBoxAttributes: object
}

export interface MoveCameraAction {
  /** Type of the action */
  type: typeof MOVE_CAMERA
  /** New position */
  newPosition: Vector3Type
}

export interface MoveCameraAndTargetAction {
  /** Type of the action */
  type: typeof MOVE_CAMERA_AND_TARGET
  /** New position */
  newPosition: Vector3Type
  /** New target */
  newTarget: Vector3Type
}

export type ActionTypes =
    InitSessionAction
    | NewItemAction
    | GoToItemAction
    | LoadItemAction
    | UpdateAllAction
    | ImageZoomAction
    | AddLabelAction
    | ChangeLabelShapeAction
    | ChangeLabelPropsAction
    | DeleteLabelAction
    | TagImageAction
    | ChangeAttributeAction
    | ChangeCategoryAction
    | ToggleAssistantViewAction
    | ChangeRectAction
    | MoveCameraAction
    | MoveCameraAndTargetAction
