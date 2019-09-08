/**
 * Define string identifiers and interfaces of actions
 */
import {
  LabelType,
  Select,
  ShapeType, Vector3Type, ViewerConfigType
} from '../functional/types'

export const INIT_SESSION = 'INIT_SESSION'
export const CHANGE_SELECT = 'CHANGE_SELECT'
export const LOAD_ITEM = 'LOAD_ITEM'
export const UPDATE_ALL = 'UPDATE_ALL'

export const IMAGE_ZOOM = 'IMAGE_ZOOM'

// Item Level
export const ADD_LABEL = 'ADD_LABEL'
export const CHANGE_LABEL_SHAPE = 'CHANGE_LABEL_SHAPE'
export const CHANGE_LABEL_PROPS = 'CHANGE_LABEL_PROPS'
export const LINK_LABELS = 'LINK_LABELS'
export const DELETE_LABEL = 'DELETE_LABEL'
// Image specific actions
export const TAG_IMAGE = 'TAG_IMAGE'

// View Level
export const TOGGLE_ASSISTANT_VIEW = 'TOGGLE_ASSISTANT_VIEW'

// Point Cloud Specific
export const MOVE_CAMERA_AND_TARGET = 'MOVE_CAMERA_AND_TARGET'

interface BaseAction {
  /** type of the action */
  type: string
  /** id of the session that initiates the action */
  sessionId: string
}

export type InitSessionAction = BaseAction

export interface ChangeSelectAction extends BaseAction {
  /** partial selection */
  select: Partial<Select>
}

export interface LoadItemAction extends BaseAction {
  /** Index of the item to load */
  itemIndex: number
  /** Configurations */
  config: ViewerConfigType
}

export type UpdateAllAction = BaseAction

export interface ImageZoomAction extends BaseAction {
  /** Zoom ratio */
  ratio: number
  /** View Offset X */
  viewOffsetX: number
  /** View Offset Y */
  viewOffsetY: number
}

export interface AddLabelAction extends BaseAction {
  /** item of the added label */
  itemIndex: number
  /** label to add */
  label: LabelType
  /** types of the shapes */
  shapeTypes: string[]
  /** Shapes of the label */
  shapes: ShapeType[]
}

export interface ChangeShapeAction extends BaseAction {
  /** item of the shape */
  itemIndex: number
  /** Shape ID */
  shapeId: number
  /** properties to update for the shape */
  props: object
}

export interface ChangeLabelAction extends BaseAction {
  /** item of the label */
  itemIndex: number
  /** Label ID */
  labelId: number
  /** properties to update for the shape */
  props: object
}

export interface LinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to link */
  labelIds: number[]
}

export interface DeleteLabelAction extends BaseAction {
  /** item of the label */
  itemIndex: number
  /** ID of label to be deleted */
  labelId: number
}

export interface TagImageAction extends BaseAction {
  /** ID of the corresponding item */
  itemIndex: number
  /** Index of the attribute */
  attributeIndex: number
  /** Index of the selected attribute */
  selectedIndex: number[]
}

export type ToggleAssistantViewAction = BaseAction

export interface MoveCameraAndTargetAction extends BaseAction {
  /** New position */
  newPosition: Vector3Type
  /** New target */
  newTarget?: Vector3Type
}

export type ActionType =
    InitSessionAction
    | ChangeSelectAction
    | LoadItemAction
    | UpdateAllAction
    | ImageZoomAction
    | AddLabelAction
    | ChangeShapeAction
    | ChangeLabelAction
    | LinkLabelsAction
    | DeleteLabelAction
    | TagImageAction
    | ToggleAssistantViewAction
    | MoveCameraAndTargetAction
