import { ActionCreator } from 'redux'
import { ThunkAction } from 'redux-thunk'
import { ReduxState } from '../common/configure_store'
import * as selector from '../common/selector'
import Session from '../common/session'
import { ConnectionStatus, IdType, LabelType,
  PaneType, Select, ShapeType, SplitType,
  TaskType, ViewerConfigType } from '../functional/types'
import * as types from './types'

/** init session */
export function initSessionAction (): types.InitSessionAction {
  return {
    type: types.INIT_SESSION,
    sessionId: Session.id
  }
}

/** update task data
 * @param {TaskType} newTask
 */
export function updateTask (newTask: TaskType): types.UpdateTaskAction {
  return {
    type: types.UPDATE_TASK,
    newTask,
    sessionId: Session.id
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

  if (Session.getState().session.trackLinking) {
    // if track linking is on, keep old labels selected
    newSelect = {
      item: index
    }
  }

  return {
    type: types.CHANGE_SELECT,
    sessionId: Session.id,
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
    type: types.CHANGE_SELECT,
    sessionId: Session.id,
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
    type: types.LOAD_ITEM,
    sessionId: Session.id,
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
    type: types.ADD_LABELS,
    sessionId: Session.id,
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
    type: types.ADD_LABELS,
    sessionId: Session.id,
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
    type: types.ADD_TRACK,
    trackType,
    sessionId: Session.id,
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
    type: types.CHANGE_SHAPES,
    sessionId: Session.id,
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
    type: types.CHANGE_SHAPES,
    sessionId: Session.id,
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
    type: types.CHANGE_SHAPES,
    sessionId: Session.id,
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
    type: types.CHANGE_LABELS,
    sessionId: Session.id,
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
    type: types.CHANGE_LABELS,
    sessionId: Session.id,
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
    type: types.LINK_LABELS,
    sessionId: Session.id,
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
    type: types.UNLINK_LABELS,
    sessionId: Session.id,
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
    type: types.MERGE_TRACKS,
    sessionId: Session.id,
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
    type: types.DELETE_LABELS,
    sessionId: Session.id,
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
    type: types.ADD_VIEWER_CONFIG,
    sessionId: Session.id,
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
    type: types.CHANGE_VIEWER_CONFIG,
    sessionId: Session.id,
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
    type: types.UPDATE_PANE,
    sessionId: Session.id,
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
    type: types.SPLIT_PANE,
    sessionId: Session.id,
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
    type: types.DELETE_PANE,
    sessionId: Session.id,
    pane,
    viewerId
  }
}

/**
 * wrapper for update all action
 */
export function updateAll (): types.UpdateAllAction {
  return {
    type: types.UPDATE_ALL,
    sessionId: Session.id
  }
}

/**
 * wrapper for submit action
 */
export function submit (): types.SubmitAction {
  return {
    type: types.SUBMIT,
    sessionId: Session.id,
    submitData: {
      time: Date.now(),
      user: Session.getState().user.id
    }
  }
}

/**
 * start to link tracks
 */
export function startLinkTrack () {
  return {
    type: types.START_LINK_TRACK,
    sessionId: Session.id
  }
}

/**
 * Update session status
 */
export function updateSessionStatus (
  status: ConnectionStatus): types.UpdateSessionStatusAction {
  return {
    type: types.UPDATE_SESSION_STATUS,
    newStatus: status,
    sessionId: Session.id
  }
}

/**
 * Mark status as reconnecting
 */
export function setStatusToReconnecting () {
  return updateSessionStatus(ConnectionStatus.RECONNECTING)
}

type ThunkCreatorType =
  ActionCreator<
  ThunkAction<void, ReduxState, void, types.ActionType>>

/**
 * Mark status as saving, unless compute is ongoing
 */
export const setStatusToSaving: ThunkCreatorType = () => {
  return (dispatch, getState) => {
    if (!selector.isStatusComputing(getState())) {
      dispatch(updateSessionStatus(ConnectionStatus.SAVING))
    }
  }
}

/**
 * Mark status as unsaved, unless some other event is in progress
 */
export const setStatusToUnsaved: ThunkCreatorType = () => {
  return (dispatch, getState) => {
    if (selector.isSessionStatusStable(getState())) {
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
  return (dispatch, getState) => {
    setTimeout(() => {
      const newNumUpdates = selector.getNumStatusUpdates(getState())
      if (numUpdates + 1 === newNumUpdates) {
        dispatch(updateSessionStatus(status))
      }
    }, 5000)
  }
}

/**
 * Mark compute done in the status
 */
export const setStatusToComputeDone: ThunkCreatorType = () => {
  return (dispatch, getState) => {
    const numUpdates = selector.getNumStatusUpdates(getState())
    dispatch(updateSessionStatus(ConnectionStatus.NOTIFY_COMPUTE_DONE))
    dispatch(
      updateSessionStatusDelayed(ConnectionStatus.COMPUTE_DONE, numUpdates)
    )
  }
}

/**
 * Mark saving as done in the status
 */
export const setStatusToSaved: ThunkCreatorType = () => {
  return (dispatch, getState) => {
    const numUpdates = selector.getNumStatusUpdates(getState())
    dispatch(updateSessionStatus(ConnectionStatus.NOTIFY_SAVED))
    dispatch(
      updateSessionStatusDelayed(ConnectionStatus.SAVED, numUpdates)
    )
  }
}
