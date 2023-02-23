import { getStateGetter } from "../common/session"
import { GetStateFunc } from "../common/simple_store"
import { uid } from "../common/uid"
import * as actionConsts from "../const/action"
import * as selector from "../functional/selector"
import * as actionTypes from "../types/action"
import { SyncActionMessageType } from "../types/message"
import { ThunkCreatorType } from "../types/redux"
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
  ShapeAllType,
  ShapeType,
  SplitType,
  State,
  TaskType,
  ViewerConfigType
} from "../types/state"

let getState = getStateGetter()

/**
 * Set the state getter for actions
 *
 * @param getter
 */
export function setActionStateGetter(getter: GetStateFunc): void {
  getState = getter
}

/**
 * Make the base action that can be extended by the other actions
 *
 * @param type
 * @param frontendOnly
 */
export function makeBaseAction(
  type: string,
  frontendOnly: boolean = false
): actionTypes.BaseAction {
  const state = getState()
  return {
    actionId: uid(),
    type,
    sessionId: state.session.id,
    userId: state.user.id,
    timestamp: Date.now(),
    frontendOnly
  }
}

/**
 * Make a null action
 *
 * @param message
 */
export function makeNullAction(message: string = ""): actionTypes.NullAction {
  return {
    ...makeBaseAction(actionConsts.NULL),
    message
  }
}

/** init session */
export function initSessionAction(): actionTypes.InitSessionAction {
  return makeBaseAction(actionConsts.INIT_SESSION)
}

/** Update task data
 *
 * @param {TaskType} newTask
 */
export function updateTask(newTask: TaskType): actionTypes.UpdateTaskAction {
  return {
    ...makeBaseAction(actionConsts.UPDATE_TASK, true),
    newTask
  }
}

/**
 * Initialize state data
 *
 * @param {TaskType} newTask
 * @param newState
 */
export function updateState(
  newState: DeepPartialState
): actionTypes.UpdateStateAction {
  return {
    ...makeBaseAction(actionConsts.UPDATE_STATE, true),
    newState
  }
}

/**
 * Go to item at index
 *
 * @param {number} index
 * @returns {actionTypes.ChangeSelectAction}
 */
export function goToItem(index: number): actionTypes.ChangeSelectAction {
  // Normally, unselect labels when item changes
  let newSelect: Partial<Select> = {
    item: index,
    labels: {},
    shapes: {}
  }

  if (getState().session.trackLinking) {
    // If track linking is on, keep old labels selected
    newSelect = {
      item: index
    }
  }

  return {
    ...makeBaseAction(actionConsts.CHANGE_SELECT),
    select: newSelect
  }
}

/**
 * Change the current selection
 *
 * @param select
 */
export function changeSelect(
  select: Partial<Select>
): actionTypes.ChangeSelectAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_SELECT),
    select
  }
}

/**
 * Create load item action
 *
 * @param itemIndex
 * @param sensorId
 */
export function loadItem(
  itemIndex: number,
  sensorId: number
): actionTypes.LoadItemAction {
  return {
    ...makeBaseAction(actionConsts.LOAD_ITEM),
    itemIndex,
    sensorId
  }
}

/**
 * Add label to the item
 *
 * @param {number} itemIndex
 * @param {LabelType} label
 * @param {ShapeType[]} shapes
 * @returns {AddLabelAction}
 */
export function addLabel(
  itemIndex: number,
  label: LabelType,
  shapes: ShapeType[] = []
): actionTypes.AddLabelsAction {
  return {
    ...makeBaseAction(actionConsts.ADD_LABELS),
    itemIndices: [itemIndex],
    labels: [[label]],
    shapes: [[shapes]]
  }
}

/**
 * Add labels to a single item
 *
 * @param itemIndex
 * @param labels
 * @param shapes
 */
export function addLabelsToItem(
  itemIndex: number,
  labels: LabelType[],
  shapes: ShapeType[][] = []
): actionTypes.AddLabelsAction {
  return {
    ...makeBaseAction(actionConsts.ADD_LABELS),
    itemIndices: [itemIndex],
    labels: [labels],
    shapes: [shapes]
  }
}

/**
 * Add a track
 *
 * @param itemIndices
 * @param trackType
 * @param labels
 * @param shapeTypes
 * @param shapes
 */
export function addTrack(
  itemIndices: number[],
  trackType: string,
  labels: LabelType[],
  shapes: ShapeType[][]
): actionTypes.AddTrackAction {
  return {
    ...makeBaseAction(actionConsts.ADD_TRACK),
    trackType,
    itemIndices,
    labels,
    shapes
  }
}

/**
 * Change the shape of the label
 *
 * @param {number} itemIndex
 * @param {number} shapeId
 * @param {Partial<ShapeType>} props
 * @param shapeIds
 * @param shapes
 * @returns {ChangeLabelShapeAction}
 */
export function changeShapes(
  itemIndex: number,
  shapeIds: IdType[],
  shapes: Array<Partial<ShapeAllType>>
): actionTypes.ChangeShapesAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_SHAPES),
    itemIndices: [itemIndex],
    shapeIds: [shapeIds],
    shapes: [shapes]
  }
}

/**
 * Change shapes in items
 *
 * @param itemIndices
 * @param shapeIds
 * @param shapes
 */
export function changeShapesInItems(
  itemIndices: number[],
  shapeIds: IdType[][],
  shapes: Array<Array<Partial<ShapeType>>>
): actionTypes.ChangeShapesAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_SHAPES),
    itemIndices,
    shapeIds,
    shapes
  }
}

/**
 * Change the shape of the label
 *
 * @param {number} itemIndex
 * @param {number} shapeId
 * @param {Partial<ShapeType>} props
 * @param shape
 * @returns {ChangeLabelShapeAction}
 */
export function changeLabelShape(
  itemIndex: number,
  shapeId: IdType,
  shape: Partial<ShapeType>
): actionTypes.ChangeShapesAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_SHAPES),
    itemIndices: [itemIndex],
    shapeIds: [[shapeId]],
    shapes: [[shape]]
  }
}

/**
 * Change the properties of the label
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>}props
 * @returns {ChangeLabelPropsAction}
 */
export function changeLabelProps(
  itemIndex: number,
  labelId: IdType,
  props: Partial<LabelType>
): actionTypes.ChangeLabelsAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_LABELS),
    itemIndices: [itemIndex],
    labelIds: [[labelId]],
    props: [[props]]
  }
}

/**
 * Change the properties of the labels
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @param itemIndices
 * @param labelIds
 * @param {Partial<LabelType>} props
 * @returns {ChangeLabelPropsAction}
 */
export function changeLabelsProps(
  itemIndices: number[],
  labelIds: IdType[][],
  props: Array<Array<Partial<LabelType>>>
): actionTypes.ChangeLabelsAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_LABELS),
    itemIndices,
    labelIds,
    props
  }
}

/**
 * Link labels
 *
 * @param {number} itemIndex
 * @param {[]number} labelIds labels to link
 */
export function linkLabels(
  itemIndex: number,
  labelIds: IdType[]
): actionTypes.LinkLabelsAction {
  return {
    ...makeBaseAction(actionConsts.LINK_LABELS),
    itemIndex,
    labelIds
  }
}

/**
 * unlink labels
 *
 * @param {number} itemIndex
 * @param {[]number} labelIds labels to unlink
 */
export function unlinkLabels(
  itemIndex: number,
  labelIds: IdType[]
): actionTypes.UnlinkLabelsAction {
  return {
    ...makeBaseAction(actionConsts.UNLINK_LABELS),
    itemIndex,
    labelIds
  }
}

/**
 * Merge tracks
 *
 * @param trackIds
 */
export function mergeTracks(trackIds: IdType[]): actionTypes.MergeTrackAction {
  return {
    ...makeBaseAction(actionConsts.MERGE_TRACKS),
    trackIds
  }
}

/**
 * Split track
 *
 * @param trackId
 * @param splitIndex
 * @param newTrackId
 */
export function splitTrack(
  trackId: IdType,
  newTrackId: IdType,
  splitIndex: number
): actionTypes.SplitTrackAction {
  return {
    ...makeBaseAction(actionConsts.SPLIT_TRACK),
    trackId,
    newTrackId,
    splitIndex
  }
}

/**
 * Delete given label
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @returns {DeleteLabelAction}
 */
export function deleteLabel(
  itemIndex: number,
  labelId: IdType
): actionTypes.DeleteLabelsAction {
  return deleteLabels([itemIndex], [[labelId]])
}

/**
 * Delete all the input labels
 *
 * @param {number[]} itemIndices
 * @param {number[][]} labelIds
 * @returns {actionTypes.DeleteLabelsAction}
 */
export function deleteLabels(
  itemIndices: number[],
  labelIds: IdType[][]
): actionTypes.DeleteLabelsAction {
  return {
    ...makeBaseAction(actionConsts.DELETE_LABELS),
    itemIndices,
    labelIds
  }
}

/**
 * Add new viewer config
 *
 * @param id
 * @param config
 */
export function addViewerConfig(
  id: number,
  config: ViewerConfigType
): actionTypes.AddViewerConfigAction {
  return {
    ...makeBaseAction(actionConsts.ADD_VIEWER_CONFIG),
    id,
    config
  }
}

/**
 * Change viewer config
 *
 * @param configs
 * @param viewerId
 * @param config
 */
export function changeViewerConfig(
  viewerId: number,
  config: ViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_VIEWER_CONFIG),
    viewerId,
    config
  }
}

/**
 * Toggle viewer config synchronization
 *
 * @param viewerId
 * @param config
 */
export function toggleSynchronization(
  viewerId: number,
  config: ViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  return changeViewerConfig(viewerId, {
    ...config,
    synchronized: !config.synchronized
  })
}

/**
 * action to updater pane
 *
 * @param pane
 * @param props
 */
export function updatePane(
  pane: number,
  props: Partial<PaneType>
): actionTypes.UpdatePaneAction {
  return {
    ...makeBaseAction(actionConsts.UPDATE_PANE),
    pane,
    props
  }
}

/**
 * action to split pane
 *
 * @param pane
 * @param split
 * @param viewerId
 */
export function splitPane(
  pane: number,
  split: SplitType,
  viewerId: number
): actionTypes.SplitPaneAction {
  return {
    ...makeBaseAction(actionConsts.SPLIT_PANE),
    pane,
    split,
    viewerId
  }
}

/**
 * action to delete pane
 *
 * @param pane
 * @param viewerId
 */
export function deletePane(
  pane: number,
  viewerId: number
): actionTypes.DeletePaneAction {
  return {
    ...makeBaseAction(actionConsts.DELETE_PANE),
    pane,
    viewerId
  }
}

/**
 * wrapper for update all action
 */
export function updateAll(): actionTypes.UpdateAllAction {
  return makeBaseAction(actionConsts.UPDATE_ALL)
}

/**
 * wrapper for submit action
 */
export function submit(): actionTypes.SubmitAction {
  return {
    ...makeBaseAction(actionConsts.SUBMIT),
    submitData: {
      time: Date.now(),
      user: getState().user.id
    }
  }
}

/**
 * start to link tracks
 */
export function startLinkTrack(): actionTypes.BaseAction {
  return makeBaseAction(actionConsts.START_LINK_TRACK)
}

/**
 * stop linking tracks
 */
export function stopLinkTrack(): actionTypes.BaseAction {
  return makeBaseAction(actionConsts.STOP_LINK_TRACK)
}

/**
 * update polygon 2d boundary clone status
 *
 * @param status
 */
export function updatePolygon2DBoundaryCloneStatus(
  status: Polygon2DBoundaryCloneStatusType | undefined
): actionTypes.UpdatePolygon2DBoundaryCloneStatusAction {
  return {
    ...makeBaseAction(actionConsts.UPDATE_POLYGON2D_BOUNDARY_CLONE),
    status
  }
}

/**
 * Finish session registration by loading backend state
 *
 * @param state
 */
export function registerSession(
  state: State
): actionTypes.RegisterSessionAction {
  return {
    ...makeBaseAction(actionConsts.REGISTER_SESSION),
    initialState: state
  }
}

/**
 * Handle broadcasted message contains one or more actions
 *
 * @param message
 */
export function receiveBroadcast(
  message: SyncActionMessageType
): actionTypes.ReceiveBroadcastAction {
  return {
    ...makeBaseAction(actionConsts.RECEIVE_BROADCAST),
    message
  }
}

/**
 * Handle session connection
 */
export function connect(): actionTypes.ConnectAction {
  return makeBaseAction(actionConsts.CONNECT)
}

/**
 * Handle session disconnection
 */
export function disconnect(): actionTypes.DisconnectAction {
  return makeBaseAction(actionConsts.DISCONNECT)
}

/**
 * Trigger save to server
 */
export function save(): actionTypes.SaveAction {
  return makeBaseAction(actionConsts.SAVE)
}

/**
 * Update session status
 *
 * @param status
 */
export function updateSessionStatus(
  status: ConnectionStatus
): actionTypes.UpdateSessionStatusAction {
  return {
    ...makeBaseAction(actionConsts.UPDATE_SESSION_STATUS),
    newStatus: status
  }
}

/**
 * Mark status as reconnecting
 */
export function setStatusToReconnecting(): actionTypes.UpdateSessionStatusAction {
  return updateSessionStatus(ConnectionStatus.RECONNECTING)
}

/**
 * Mark status as submitting
 */
export function setStatusToSubmitting(): actionTypes.UpdateSessionStatusAction {
  return updateSessionStatus(ConnectionStatus.SUBMITTING)
}

/**
 * Mark status as saving, unless compute is ongoing
 */
export const setStatusToSaving: ThunkCreatorType = () => {
  return (dispatch, localGetState) => {
    if (!selector.isStatusComputing(localGetState())) {
      dispatch(updateSessionStatus(ConnectionStatus.SAVING))
    }
  }
}

/**
 * Mark status as unsaved, unless some other event is in progress
 */
export const setStatusToUnsaved: ThunkCreatorType = () => {
  return (dispatch, localGetState) => {
    if (selector.isSessionStatusStable(localGetState())) {
      dispatch(updateSessionStatus(ConnectionStatus.UNSAVED))
    }
  }
}

/**
 * After a connect/reconnect, mark status as unsaved
 * Regardless of previous status
 */
export function setStatusAfterConnect(): actionTypes.UpdateSessionStatusAction {
  return updateSessionStatus(ConnectionStatus.UNSAVED)
}

/**
 * Mark status as computing
 */
export function setStatusToComputing(): actionTypes.UpdateSessionStatusAction {
  return updateSessionStatus(ConnectionStatus.COMPUTING)
}

/**
 * Change Overlays that are displayed
 */
export function changeOverlays(status:number[]): actionTypes.ChangeOverlaysAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_OVERLAYS),
    newOverlayStatus: status
  }
}

/**
 * Change Overlay Transparency
 */
export function changeOverlayTransparency(transparency:number): actionTypes.ChangeOverlayTransparencyAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_OVERLAY_TRANSPARENCY),
    newOverlayTransparency: transparency
  }
} 

/**
 * Change Radar Status
 */
export function changeRadarStatus(status:number[]): actionTypes.ChangeRadarStatusAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_RADAR_STATUS),
    newRadarStatus: status
  }
}


/**
 * Change session mode
 *
 * @param mode
 */
export function changeSessionMode(
  mode: ModeStatus
): actionTypes.ChangeSessionModeAction {
  return {
    ...makeBaseAction(actionConsts.CHANGE_SESSION_MODE),
    newMode: mode
  }
}

/**
 * Mark mode as selecting
 */
export function changeModeToSelecting(): actionTypes.ChangeSessionModeAction {
  return changeSessionMode(ModeStatus.SELECTING)
}

/**
 * Mark mode as selecting
 */
export function changeModeToAnnotating(): actionTypes.ChangeSessionModeAction {
  return changeSessionMode(ModeStatus.ANNOTATING)
}

/**
 * After 5 seconds, fade out the previous message
 * If no other actions occurred in the meantime
 *
 * @param status
 * @param numUpdates
 */
export const updateSessionStatusDelayed: ThunkCreatorType = (
  status: ConnectionStatus,
  numUpdates: number
) => {
  return (dispatch, localGetState) => {
    setTimeout(() => {
      const newNumUpdates = selector.getNumStatusUpdates(localGetState())
      if (numUpdates + 1 === newNumUpdates) {
        dispatch(updateSessionStatus(status))
      }
    }, 5000)
  }
}

/**
 * Update submission banner and trigger fadeout animation
 *
 * @param notifyStatus
 * @param fadeStatus
 */
export const setStatusForBanner: ThunkCreatorType = (
  notifyStatus: ConnectionStatus,
  fadeStatus: ConnectionStatus
) => {
  return (dispatch, localGetState) => {
    const numUpdates = selector.getNumStatusUpdates(localGetState())
    dispatch(updateSessionStatus(notifyStatus))
    dispatch(updateSessionStatusDelayed(fadeStatus, numUpdates))
  }
}

/**
 * Mark compute done in the status
 */
export const setStatusToComputeDone: ThunkCreatorType = () => {
  return setStatusForBanner(
    ConnectionStatus.NOTIFY_COMPUTE_DONE,
    ConnectionStatus.COMPUTE_DONE
  )
}

/**
 * Mark saving as done in the status
 */
export const setStatusToSaved: ThunkCreatorType = () => {
  return setStatusForBanner(
    ConnectionStatus.NOTIFY_SAVED,
    ConnectionStatus.SAVED
  )
}

/**
 * Mark submitting as done in the status
 */
export const setStatusToSubmitted: ThunkCreatorType = () => {
  return setStatusForBanner(
    ConnectionStatus.NOTIFY_SUBMITTED,
    ConnectionStatus.SUBMITTED
  )
}

/**
 * Merge actions into an sequential action
 *
 * @param actions
 * @param removeNull
 */
export function makeSequential(
  actions: actionTypes.BaseAction[],
  removeNull: boolean = false
): actionTypes.SequentialAction {
  if (removeNull) {
    actions = actions.filter((a) => a.type !== actionConsts.NULL)
  }
  return {
    ...makeBaseAction(actionConsts.SEQUENTIAL),
    actions
  }
}

/**
 * Add alert
 *
 * @param alert
 */
export function addAlert(alert: AlertType): actionTypes.AddAlertAction {
  return {
    ...makeBaseAction(actionConsts.ADD_ALERT),
    alert
  }
}

/**
 * Close alert
 *
 * @param alertId
 */
export function removeAlert(alertId: string): actionTypes.RemoveAlertAction {
  return {
    ...makeBaseAction(actionConsts.CLOSE_ALERT),
    alertId
  }
}
