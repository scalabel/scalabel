import { ActionCreator } from 'redux'
import { ThunkAction } from 'redux-thunk'
import { ReduxState } from '../common/configure_store'
import { getStateGetter } from '../common/session'
import { getStateFunc } from '../common/simple_store'
import { uid } from '../common/uid'
import * as selector from '../functional/selector'
import { ConnectionStatus, IdType, LabelType,
  PaneType, Select, ShapeType, SplitType,
  TaskType, ViewerConfigType } from '../functional/types'
import * as types from './types'

let getState = getStateGetter()

/**
 * Set the state getter for actions
 * @param getter
 */
export function setActionStateGetter (getter: getStateFunc) {
  getState = getter
}

/**
 * Make the base action that can be extended by the other actions
 * @param type
 */
export function makeBaseAction (type: string): types.BaseAction {
  const state = getState()
  return {
    actionId: uid(),
    type,
    sessionId: state.session.id,
    userId: state.user.id,
    timestamp: Date.now()
  }
}

/** init session */
export function initSessionAction (): types.InitSessionAction {
  return makeBaseAction(types.INIT_SESSION)
}

/** update task data
 * @param {TaskType} newTask
 */
export function updateTask (newTask: TaskType): types.UpdateTaskAction {
  return {
    ...makeBaseAction(types.UPDATE_TASK),
    newTask
  }
}

/**
 * Go to item at index
 * @param {number} index
 * @return {types.ChangeSelectAction}
 */
export function goToItem (index: number): types.ChangeSelectAction {
  // normally, unselect labels when item changes
  let newSelect: Partial<Select> = {
    item: index,
    labels: {},
    shapes: {}
  }

  if (getState().session.trackLinking) {
    // if track linking is on, keep old labels selected
    newSelect = {
      item: index
    }
  }

  return {
    ...makeBaseAction(types.CHANGE_SELECT),
    select: newSelect
  }
}

/**
 * Change the current selection
 * @param select
 */
export function changeSelect (
    select: Partial<Select>): types.ChangeSelectAction {
  return {
    ...makeBaseAction(types.CHANGE_SELECT),
    select
  }
}

/**
 * Create load item action
 */
export function loadItem (
  itemIndex: number,
  sensorId: number
): types.LoadItemAction {
  return {
    ...makeBaseAction(types.LOAD_ITEM),
    itemIndex,
    sensorId
  }
}

/**
 * Add label to the item
 * @param {number} itemIndex
 * @param {LabelType} label
 * @param {ShapeType[]} shapes
 * @return {AddLabelAction}
 */
export function addLabel (
  itemIndex: number,
  label: LabelType,
  shapes: ShapeType[] = []
): types.AddLabelsAction {
  return {
    ...makeBaseAction(types.ADD_LABELS),
    itemIndices: [itemIndex],
    labels: [[label]],
    shapes: [[shapes]]
  }
}

/**
 * Add labels to a single item
 * @param itemIndex
 * @param labels
 * @param shapes
 */
export function addLabelsToItem (
  itemIndex: number,
  labels: LabelType[],
  shapes: ShapeType[][] = []
): types.AddLabelsAction {
  return {
    ...makeBaseAction(types.ADD_LABELS),
    itemIndices: [itemIndex],
    labels: [labels],
    shapes: [shapes]
  }
}

/**
 * Add a track
 * @param itemIndices
 * @param labels
 * @param shapeTypes
 * @param shapes
 */
export function addTrack (
  itemIndices: number[],
  trackType: string,
  labels: LabelType[],
  shapes: ShapeType[][]
): types.AddTrackAction {
  return {
    ...makeBaseAction(types.ADD_TRACK),
    trackType,
    itemIndices,
    labels,
    shapes
  }
}

/**
 * Change the shape of the label
 * @param {number} itemIndex
 * @param {number} shapeId
 * @param {Partial<ShapeType>} props
 * @return {ChangeLabelShapeAction}
 */
export function changeShapes (
    itemIndex: number, shapeIds: IdType[], shapes: Array<Partial<ShapeType>>
  ): types.ChangeShapesAction {
  return {
    ...makeBaseAction(types.CHANGE_SHAPES),
    itemIndices: [itemIndex],
    shapeIds: [shapeIds],
    shapes: [shapes]
  }
}

/**
 * Change shapes in items
 * @param itemIndices
 * @param shapeIds
 * @param shapes
 */
export function changeShapesInItems (
  itemIndices: number[], shapeIds: IdType[][],
  shapes: Array<Array<Partial<ShapeType>>>
): types.ChangeShapesAction {
  return {
    ...makeBaseAction(types.CHANGE_SHAPES),
    itemIndices,
    shapeIds,
    shapes
  }
}

/**
 * Change the shape of the label
 * @param {number} itemIndex
 * @param {number} shapeId
 * @param {Partial<ShapeType>} props
 * @return {ChangeLabelShapeAction}
 */
export function changeLabelShape (
    itemIndex: number, shapeId: IdType, shape: Partial<ShapeType>
  ): types.ChangeShapesAction {
  return {
    ...makeBaseAction(types.CHANGE_SHAPES),
    itemIndices: [itemIndex],
    shapeIds: [[shapeId]],
    shapes: [[shape]]
  }
}

/**
 * Change the properties of the label
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>}props
 * @return {ChangeLabelPropsAction}
 */
export function changeLabelProps (
    itemIndex: number, labelId: IdType, props: Partial<LabelType>
  ): types.ChangeLabelsAction {
  return {
    ...makeBaseAction(types.CHANGE_LABELS),
    itemIndices: [itemIndex],
    labelIds: [[labelId]],
    props: [[props ]]
  }
}

/**
 * Change the properties of the labels
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>}props
 * @return {ChangeLabelPropsAction}
 */
export function changeLabelsProps (
  itemIndices: number[], labelIds: IdType[][],
  props: Array<Array<Partial<LabelType>>>
): types.ChangeLabelsAction {
  return {
    ...makeBaseAction(types.CHANGE_LABELS),
    itemIndices,
    labelIds,
    props
  }
}

/**
 * Link labels
 * @param {number} itemIndex
 * @param {[]number} labelIds labels to link
 */
export function linkLabels (
    itemIndex: number, labelIds: IdType[]): types.LinkLabelsAction {
  return {
    ...makeBaseAction(types.LINK_LABELS),
    itemIndex,
    labelIds
  }
}

/**
 * unlink labels
 * @param {number} itemIndex
 * @param {[]number} labelIds labels to unlink
 */
export function unlinkLabels (
    itemIndex: number, labelIds: IdType[]): types.UnlinkLabelsAction {
  return {
    ...makeBaseAction(types.UNLINK_LABELS),
    itemIndex,
    labelIds
  }
}

/**
 * Merge tracks
 * @param trackIds
 */
export function mergeTracks (trackIds: IdType[]): types.MergeTrackAction {
  return {
    ...makeBaseAction(types.MERGE_TRACKS),
    trackIds
  }
}

/**
 * Delete given label
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {DeleteLabelAction}
 */
export function deleteLabel (
    itemIndex: number, labelId: IdType): types.DeleteLabelsAction {
  return deleteLabels([itemIndex], [[labelId]])
}

/**
 * Delete all the input labels
 * @param {number[]} itemIndices
 * @param {number[][]} labelIds
 * @return {types.DeleteLabelsAction}
 */
export function deleteLabels (
  itemIndices: number[], labelIds: IdType[][]): types.DeleteLabelsAction {
  return {
    ...makeBaseAction(types.DELETE_LABELS),
    itemIndices,
    labelIds
  }
}

/**
 * Add new viewer config
 * @param config
 */
export function addViewerConfig (
  id: number,
  config: ViewerConfigType
): types.AddViewerConfigAction {
  return {
    ...makeBaseAction(types.ADD_VIEWER_CONFIG),
    id,
    config
  }
}

/**
 * Change viewer config
 * @param configs
 */
export function changeViewerConfig (
  viewerId: number, config: ViewerConfigType
): types.ChangeViewerConfigAction {
  return {
    ...makeBaseAction(types.CHANGE_VIEWER_CONFIG),
    viewerId,
    config
  }
}

/** Toggle viewer config synchronization */
export function toggleSynchronization (
  viewerId: number, config: ViewerConfigType
) {
  return changeViewerConfig(
    viewerId,
    { ...config, synchronized: !config.synchronized }
  )
}

/** action to updater pane */
export function updatePane (
  pane: number,
  props: Partial<PaneType>
): types.UpdatePaneAction {
  return {
    ...makeBaseAction(types.UPDATE_PANE),
    pane,
    props
  }
}

/** action to split pane */
export function splitPane (
  pane: number,
  split: SplitType,
  viewerId: number
): types.SplitPaneAction {
  return {
    ...makeBaseAction(types.SPLIT_PANE),
    pane,
    split,
    viewerId
  }
}

/** action to delete pane */
export function deletePane (
  pane: number,
  viewerId: number
): types.DeletePaneAction {
  return {
    ...makeBaseAction(types.DELETE_PANE),
    pane,
    viewerId
  }
}

/**
 * wrapper for update all action
 */
export function updateAll (): types.UpdateAllAction {
  return makeBaseAction(types.UPDATE_ALL)
}

/**
 * wrapper for submit action
 */
export function submit (): types.SubmitAction {
  return {
    ...makeBaseAction(types.SUBMIT),
    submitData: {
      time: Date.now(),
      user: getState().user.id
    }
  }
}

/**
 * start to link tracks
 */
export function startLinkTrack () {
  return makeBaseAction(types.START_LINK_TRACK)
}

/**
 * Update session status
 */
export function updateSessionStatus (
  status: ConnectionStatus): types.UpdateSessionStatusAction {
  return {
    ...makeBaseAction(types.UPDATE_SESSION_STATUS),
    newStatus: status
  }
}

/**
 * Mark status as reconnecting
 */
export function setStatusToReconnecting () {
  return updateSessionStatus(ConnectionStatus.RECONNECTING)
}

/**
 * Mark status as submitting
 */
export function setStatusToSubmitting () {
  return updateSessionStatus(ConnectionStatus.SUBMITTING)
}

type ThunkCreatorType =
  ActionCreator<
  ThunkAction<void, ReduxState, void, types.ActionType>>

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
export function setStatusAfterConnect () {
  return updateSessionStatus(ConnectionStatus.UNSAVED)
}

/**
 * Mark status as computing
 */
export function setStatusToComputing () {
  return updateSessionStatus(ConnectionStatus.COMPUTING)
}

/**
 * After 5 seconds, fade out the previous message
 * If no other actions occurred in the meantime
 */
export const updateSessionStatusDelayed: ThunkCreatorType = (
  status: ConnectionStatus, numUpdates: number) => {
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
 */
export const setStatusForBanner: ThunkCreatorType = (
  notifyStatus: ConnectionStatus, fadeStatus: ConnectionStatus
) => {
  return (dispatch, localGetState) => {
    const numUpdates = selector.getNumStatusUpdates(localGetState())
    dispatch(updateSessionStatus(notifyStatus))
    dispatch(
      updateSessionStatusDelayed(fadeStatus, numUpdates)
    )
  }
}

/**
 * Mark compute done in the status
 */
export const setStatusToComputeDone: ThunkCreatorType = () => {
  return setStatusForBanner(ConnectionStatus.NOTIFY_COMPUTE_DONE,
    ConnectionStatus.COMPUTE_DONE)
}

/**
 * Mark saving as done in the status
 */
export const setStatusToSaved: ThunkCreatorType = () => {
  return setStatusForBanner(ConnectionStatus.NOTIFY_SAVED,
    ConnectionStatus.SAVED)
}

/**
 * Mark submitting as done in the status
 */
export const setStatusToSubmitted: ThunkCreatorType = () => {
  return setStatusForBanner(ConnectionStatus.NOTIFY_SUBMITTED,
    ConnectionStatus.SUBMITTED)
}
