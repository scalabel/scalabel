/**
 * Define string identifiers and interfaces of actions
 */
import { SyncActionMessageType } from "./message"
import {
  AlertType,
  ConnectionStatus,
  DeepPartialState,
  IdType,
  LabelType,
  ModeStatus,
  PaneType,
  Polygon2DBoundaryCloneStatusType,
  Select,
  ShapeType,
  SplitType,
  State,
  SubmitData,
  TaskType,
  ViewerConfigType
} from "./state"

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

export interface NullAction extends BaseAction {
  /** an optional message describing the null action */
  message: string
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
  /** Task data to use */
  newTask: TaskType
}

export interface UpdateStateAction extends BaseAction {
  /** Initial state data */
  newState: DeepPartialState
}

export interface UpdateSessionStatusAction extends BaseAction {
  /** New status of the session */
  newStatus: ConnectionStatus
}

export interface ChangeOverlaysAction extends BaseAction {
  /** New overlays to activate */
  newOverlayStatus: number[] 
}

export interface ChangeSessionModeAction extends BaseAction {
  /** New mode of the session */
  newMode: ModeStatus
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

export interface SplitTrackAction extends BaseAction {
  /** item of the added label */
  trackId: IdType
  newTrackId: IdType
  splitIndex: number
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
  itemIndex: number
  /** ids of the labels to link */
  labelIds: IdType[]
}

export interface UnlinkLabelsAction extends BaseAction {
  /** item of the labels */
  itemIndex: number
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
  /** ID of pane to delete */
  pane: number
  /** ID of corresponding viewer config */
  viewerId: number
}

export interface RegisterSessionAction extends BaseAction {
  /** Initial state received from the backend */
  initialState: State
}

export interface ReceiveBroadcastAction extends BaseAction {
  /** The message containing the broadcasted action/actions */
  message: SyncActionMessageType
}

export type ConnectAction = BaseAction

export type DisconnectAction = BaseAction

export type SaveAction = BaseAction

export interface SequentialAction extends BaseAction {
  /** actions in this sequence */
  actions: BaseAction[]
}

export interface AddAlertAction extends BaseAction {
  /** id of the alert to close */
  alert: AlertType
}

export interface RemoveAlertAction extends BaseAction {
  /** id of the alert to close */
  alertId: string
}

export interface UpdatePolygon2DBoundaryCloneStatusAction extends BaseAction {
  /** current interaction status */
  status: Polygon2DBoundaryCloneStatusType | undefined
}

export type ActivateSpanAction = BaseAction

export type DeactivateSpanAction = BaseAction

export type UpdateSpanPoint = BaseAction

export type RegisterSpanPointAction = BaseAction

export type ResetSpanAction = BaseAction

export type PauseSpanAction = BaseAction

export type ResumeSpanAction = BaseAction

export type UndoSpanAction = BaseAction

export type ToggleGroundPlaneAction = BaseAction

/**
 * These actions are event-driven messages intercepted by the sync middleware
 */
export type SyncActionType =
  | RegisterSessionAction
  | ReceiveBroadcastAction
  | ConnectAction
  | DisconnectAction
  | SaveAction

export type SessionActionType =
  | InitSessionAction
  | LoadItemAction
  | UpdateAllAction
  | UpdateTaskAction
  | UpdateStateAction
  | UpdateSessionStatusAction
  | AddAlertAction
  | RemoveAlertAction
  | SyncActionType
  | UpdatePolygon2DBoundaryCloneStatusAction

export type Info3DActionType =
  | ActivateSpanAction
  | DeactivateSpanAction
  | RegisterSpanPointAction
  | ResetSpanAction
  | PauseSpanAction
  | ResumeSpanAction
  | UndoSpanAction
  | ToggleGroundPlaneAction

export type UserActionType =
  | ChangeSelectAction
  | ChangeViewerConfigAction
  | AddViewerConfigAction
  | UpdatePaneAction
  | SplitPaneAction
  | DeletePaneAction

export type TaskActionType =
  | AddLabelsAction
  | ChangeShapesAction
  | ChangeLabelsAction
  | DeleteLabelsAction
  | LinkLabelsAction
  | AddTrackAction
  | MergeTrackAction
  | SubmitAction

export type ActionType =
  | SessionActionType
  | UserActionType
  | TaskActionType
  | SequentialAction
