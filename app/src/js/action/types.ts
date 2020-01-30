/**
 * Define string identifiers and interfaces of actions
 */
import {
  PaneType,
  PartialIndexedShapeType,
  PartialLabelType,
  Select,
  SplitType,
  TaskType,
  TrackType,
  ViewerConfigType
} from '../functional/types'

export const INIT_SESSION = 'INIT_SESSION'
export const CHANGE_SELECT = 'CHANGE_SELECT'
export const LOAD_ITEM = 'LOAD_ITEM'
export const UPDATE_ALL = 'UPDATE_ALL'
export const UPDATE_TASK = 'UPDATE_TASK'

// Item Level
export const UPDATE_LABELS = 'UPDATE_LABELS'
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

export const TASK_ACTION_TYPES = [
  UPDATE_LABELS,
  LINK_LABELS,
  DELETE_LABELS,
  ADD_TRACK,
  MERGE_TRACKS,
  UPDATE_TASK
]

export interface BaseAction {
  /** type of the action */
  type: string
  /** id of the session that initiates the action */
  sessionId: string
  /** timestamp given by backend. It is Date.now() */
  timestamp?: number
}

export type InitSessionAction = BaseAction

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

export interface UpdateLabelsAction extends BaseAction {
  /** Item indices */
  itemIndices: number[]
  /** Labels to be updated per item. Negative id means new label */
  labels: PartialLabelType[][]
  /** Shapes to be updated per item. Negative id means new shape */
  indexedShapes: PartialIndexedShapeType[][]
  /** Tracks to be added */
  newTracks: TrackType[]
}

export interface MergeTrackAction extends BaseAction {
  /** item of the added label */
  trackIds: number[]
}

export interface LinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to link */
  labelIds: number[]
}

export interface UnlinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number,
  /** ids of the labels to unlink */
  labelIds: number[]
}

export interface DeleteLabelsAction extends BaseAction {
  /** item of the label */
  itemIndices: number[]
  /** ID of label to be deleted */
  labelIds: number[][]
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

export type UserActionType =
  ChangeSelectAction
  | ChangeViewerConfigAction
  | AddViewerConfigAction
  | UpdatePaneAction
  | SplitPaneAction
  | DeletePaneAction

export type TaskActionType =
  UpdateLabelsAction
  | DeleteLabelsAction
  | LinkLabelsAction
  | MergeTrackAction

export type ActionType =
  SessionActionType
  | UserActionType
  | TaskActionType
