/**
 * Define string identifiers and interfaces of actions
 */
import {
  ConnectionStatus,
  IdType,
  LabelType,
  PaneType,
  Select,
  ShapeType,
  SplitType,
  SubmitData,
  TaskType,
  ViewerConfigType
} from '../functional/types'

export const INIT_SESSION = 'INIT_SESSION'
export const CHANGE_SELECT = 'CHANGE_SELECT'
export const LOAD_ITEM = 'LOAD_ITEM'
export const UPDATE_ALL = 'UPDATE_ALL'
export const UPDATE_TASK = 'UPDATE_TASK'
export const SUBMIT = 'SUBMIT'
export const UPDATE_SESSION_STATUS = 'UPDATE_SESSION_STATUS'

// Item Level
export const ADD_LABELS = 'ADD_LABELS'
export const CHANGE_SHAPES = 'CHANGE_SHAPES'
export const CHANGE_LABELS = 'CHANGE_LABELS'
export const LINK_LABELS = 'LINK_LABELS'
export const UNLINK_LABELS = 'UNLINK_LABELS'
export const DELETE_LABELS = 'DELETE_LABELS'

export const ADD_TRACK = 'ADD_TRACK'
export const MERGE_TRACKS = 'MERGE_TRACKS'

// View Level
export const ADD_VIEWER_CONFIG = 'ADD_VIEWER_CONFIG'
export const CHANGE_VIEWER_CONFIG = 'CHANGE_VIEWER_CONFIG'
export const UPDATE_PANE = 'UPDATE_PANE'
export const SPLIT_PANE = 'SPLIT_PANE'
export const DELETE_PANE = 'DELETE_PANE'
export const START_LINK_TRACK = 'START_LINK_TRACK'

/**
 * These are actions that should be shared between sessions/users
 * UPDATE_TASK deliberately not included because its used for local updates
 */
const TASK_ACTION_TYPES = [
  ADD_LABELS,
  CHANGE_SHAPES,
  CHANGE_LABELS,
  LINK_LABELS,
  DELETE_LABELS,
  ADD_TRACK,
  MERGE_TRACKS,
  SUBMIT
]

/**
 * Checks if the action modifies task
 */
export function isTaskAction (action: BaseAction) {
  return TASK_ACTION_TYPES.includes(action.type)
}

/**
 * Checks if the action list contains a submit action
 */
export function hasSubmitAction (actions: BaseAction[]): boolean {
  for (const action of actions) {
    if (action.type === SUBMIT) {
      return true
    }
  }
  return false
}

/**
 * These are actions that should not be broadcast beyond the session
 */
const SESSION_ACTION_TYPES = [
  UPDATE_SESSION_STATUS,
  CHANGE_SELECT
]

/**
 * Checks if the action modifies session
 */
export function isSessionAction (action: BaseAction) {
  return SESSION_ACTION_TYPES.includes(action.type)
}

export interface BaseAction {
  /** unique id for the action */
  actionId: IdType
  /** type of the action */
  type: string
  /** id of the session that initiates the action */
  sessionId: IdType
  /** timestamp given by backend. It is Date.now() */
  timestamp: number
  /** whether to sync action, or just apply to frontend */
  frontendOnly?: boolean
  /** id of the user that initiates the action */
  userId: IdType
}

export type InitSessionAction = BaseAction

export interface SubmitAction extends BaseAction {
  /** the data for the submission */
  submitData: SubmitData
}

export interface ChangeSelectAction extends BaseAction {
  /** partial selection */
  select: Partial<Select>
}

export interface LoadItemAction extends BaseAction {
  /** Index of the item to load */
  itemIndex: number
  /** Id of corresponding data source of element in item */
  sensorId: number
}

export type UpdateAllAction = BaseAction

export interface UpdateTaskAction extends BaseAction {
  /** task data to use */
  newTask: TaskType
}

export interface UpdateSessionStatusAction extends BaseAction {
  /** New status of the session */
  newStatus: ConnectionStatus
}

export interface AddLabelsAction extends BaseAction {
  /** item of the added label */
  itemIndices: number[]
  /** labels to add to each item */
  labels: LabelType[][]
  /** shapes for each label */
  shapes: ShapeType[][][]
}

export interface AddTrackAction extends BaseAction {
  /** track type */
  trackType: string
  /** item of the added label */
  itemIndices: number[]
  /** labels to add to each item */
  labels: LabelType[]
  /** shapes for each label */
  shapes: ShapeType[][]
}

export interface MergeTrackAction extends BaseAction {
  /** item of the added label */
  trackIds: IdType[]
}

export interface ChangeShapesAction extends BaseAction {
  /** item of the shape */
  itemIndices: number[]
  /** Shape ids in each item */
  shapeIds: IdType[][]
  /** properties to update for the shape */
  shapes: Array<Array<Partial<ShapeType>>>
}

export interface ChangeLabelsAction extends BaseAction {
  /** item of the label */
  itemIndices: number[]
  /** Label ID */
  labelIds: IdType[][]
  /** properties to update for the shape */
  props: Array<Array<Partial<LabelType>>>
}

export interface LinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to link */
  labelIds: IdType[]
}

export interface UnlinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to unlink */
  labelIds: IdType[]
}

export interface DeleteLabelsAction extends BaseAction {
  /** item of the label */
  itemIndices: number[]
  /** ID of label to be deleted */
  labelIds: IdType[][]
}

export interface AddViewerConfigAction extends BaseAction {
  /** viewer id */
  id: number
  /** config to add */
  config: ViewerConfigType
}

export interface ChangeViewerConfigAction extends BaseAction {
  /** id of viewer to update */
  viewerId: number
  /** configs to update */
  config: ViewerConfigType
}

export interface DeleteViewerConfigAction extends BaseAction {
  /** id of config to delete */
  viewerId: number
}

export interface UpdatePaneAction extends BaseAction {
  /** pane id */
  pane: number
  /** updated properties */
  props: Partial<PaneType>
}

export interface SplitPaneAction extends BaseAction {
  /** ID of pane to split */
  pane: number
  /** ID of corresponding viewer config */
  viewerId: number
  /** Split direction */
  split: SplitType
}

export interface DeletePaneAction extends BaseAction {
  /** ID of pane to split */
  pane: number
  /** ID of corresponding viewer config */
  viewerId: number
}

export type SessionActionType =
  InitSessionAction
  | LoadItemAction
  | UpdateAllAction
  | UpdateTaskAction
  | UpdateSessionStatusAction

export type UserActionType =
  ChangeSelectAction
  | ChangeViewerConfigAction
  | AddViewerConfigAction
  | UpdatePaneAction
  | SplitPaneAction
  | DeletePaneAction

export type TaskActionType =
  AddLabelsAction
  | ChangeShapesAction
  | ChangeLabelsAction
  | DeleteLabelsAction
  | LinkLabelsAction
  | AddTrackAction
  | MergeTrackAction
  | SubmitAction

export type ActionType =
  SessionActionType
  | UserActionType
  | TaskActionType
