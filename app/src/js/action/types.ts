/**
 * Define string identifiers and interfaces of actions
 */
import {
  ImageViewerConfigType,
  LabelType,
  PointCloudViewerConfigType,
  Select,
  ShapeType,
  TaskType,
  ViewerConfigType
} from '../functional/types'

export const INIT_SESSION = 'INIT_SESSION'
export const CHANGE_SELECT = 'CHANGE_SELECT'
export const LOAD_ITEM = 'LOAD_ITEM'
export const UPDATE_ALL = 'UPDATE_ALL'
export const UPDATE_TASK = 'UPDATE_TASK'

// Item Level
export const ADD_LABELS = 'ADD_LABELS'
export const CHANGE_SHAPES = 'CHANGE_SHAPES'
export const CHANGE_LABELS = 'CHANGE_LABELS'
export const LINK_LABELS = 'LINK_LABELS'
export const DELETE_LABELS = 'DELETE_LABELS'

export const ADD_TRACK = 'ADD_TRACK'
export const MERGE_TRACKS = 'MERGE_TRACKS'

// Image specific actions
export const UPDATE_IMAGE_VIEWER_CONFIG = 'UPDATE_IMAGE_VIEWER_CONFIG'

// View Level
export const TOGGLE_ASSISTANT_VIEW = 'TOGGLE_ASSISTANT_VIEW'

// Point Cloud Specific
export const UPDATE_POINT_CLOUD_VIEWER_CONFIG =
  'UPDATE_POINT_CLOUD_VIEWER_CONFIG'

export const TASK_ACTION_TYPES = [
  ADD_LABELS,
  CHANGE_SHAPES,
  CHANGE_LABELS,
  LINK_LABELS,
  DELETE_LABELS,
  ADD_TRACK,
  MERGE_TRACKS
]

export interface BaseAction {
  /** type of the action */
  type: string
  /** id of the session that initiates the action */
  sessionId: string
  /** timestamp given by backend */
  timestamp?: string
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

export interface UpdateTaskAction extends BaseAction {
  /** task data to use */
  newTask: TaskType
}

export interface UpdateImageViewerConfigAction extends BaseAction {
  /** fields to update */
  newFields: Partial<ImageViewerConfigType>
}

export interface AddLabelsAction extends BaseAction {
  /** item of the added label */
  itemIndices: number[]
  /** labels to add to each item */
  labels: LabelType[][]
  /** shape types for each label */
  shapeTypes: string[][][]
  /** shapes for each label */
  shapes: ShapeType[][][]
}

export interface AddTrackAction extends BaseAction {
  /** item of the added label */
  itemIndices: number[]
  /** labels to add to each item */
  labels: LabelType[]
  /** shape types for each label */
  shapeTypes: string[][]
  /** shapes for each label */
  shapes: ShapeType[][]
}

export interface MergeTrackAction extends BaseAction {
  /** item of the added label */
  trackIds: number[]
}

export interface ChangeShapesAction extends BaseAction {
  /** item of the shape */
  itemIndices: number[]
  /** Shape ids in each item */
  shapeIds: number[][]
  /** properties to update for the shape */
  shapes: Array<Array<Partial<ShapeType>>>
}

export interface ChangeLabelsAction extends BaseAction {
  /** item of the label */
  itemIndices: number[]
  /** Label ID */
  labelIds: number[][]
  /** properties to update for the shape */
  props: Array<Array<Partial<LabelType>>>
}

export interface LinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to link */
  labelIds: number[]
}

export interface DeleteLabelsAction extends BaseAction {
  /** item of the label */
  itemIndices: number[]
  /** ID of label to be deleted */
  labelIds: number[][]
}

export type ToggleAssistantViewAction = BaseAction

export interface UpdatePointCloudViewerConfigAction extends BaseAction {
  /** Fields to update */
  newFields: Partial<PointCloudViewerConfigType>
}

export type SessionActionType =
  InitSessionAction
  | LoadItemAction
  | UpdateAllAction
  | UpdateTaskAction

export type UserActionType =
  ChangeSelectAction
  | ToggleAssistantViewAction
  | UpdateImageViewerConfigAction
  | UpdatePointCloudViewerConfigAction

export type TaskActionType =
  AddLabelsAction
  | ChangeShapesAction
  | ChangeLabelsAction
  | DeleteLabelsAction
  | LinkLabelsAction
  | AddTrackAction
  | MergeTrackAction

export type ActionType =
  SessionActionType
  | UserActionType
  | TaskActionType
