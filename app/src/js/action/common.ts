import Session from '../common/session'
import { IndexedShapeType, LabelType,
  PaneType, PartialIndexedShapeType, PartialLabelType, Select, SplitType, TaskType, TrackType, ViewerConfigType } from '../functional/types'
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

  if (Session.trackLinking) {
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
  shapes: IndexedShapeType[] = []
): types.UpdateLabelsAction {
  return {
    type: types.UPDATE_LABELS,
    sessionId: Session.id,
    itemIndices: [itemIndex],
    labels: [[label]],
    indexedShapes: [shapes],
    newTracks: []
  }
}

/** Change labels in item */
export function changeLabels (
  itemIndex: number,
  updates: PartialLabelType[]
): types.UpdateLabelsAction {
  return {
    type: types.UPDATE_LABELS,
    sessionId: Session.id,
    itemIndices: [itemIndex],
    labels: [updates],
    indexedShapes: [],
    newTracks: []
  }
}

/** Change existing label */
export function changeLabel (
  itemIndex: number,
  updates: PartialLabelType
): types.UpdateLabelsAction {
  return {
    type: types.UPDATE_LABELS,
    sessionId: Session.id,
    itemIndices: [itemIndex],
    labels: [[updates]],
    indexedShapes: [],
    newTracks: []
  }
}

/** Add/Update labels, shapes, tracks */
export function updateLabelsShapesTracks (
  itemIndices: number[],
  labels: PartialLabelType[][],
  indexedShapes: PartialIndexedShapeType[][],
  newTracks: TrackType[]
): types.UpdateLabelsAction {
  return {
    type: types.UPDATE_LABELS,
    sessionId: Session.id,
    itemIndices,
    labels,
    indexedShapes,
    newTracks
  }
}

/**
 * Link labels
 * @param {number} itemIndex
 * @param {[]number} labelIds labels to link
 */
export function linkLabels (
    itemIndex: number, labelIds: number[]): types.LinkLabelsAction {
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
    itemIndex: number, labelIds: number[]): types.UnlinkLabelsAction {
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
export function mergeTracks (trackIds: number[]): types.MergeTrackAction {
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
    itemIndex: number, labelId: number): types.DeleteLabelsAction {
  return deleteLabels([itemIndex], [[labelId]])
}

/**
 * Delete all the input labels
 * @param {number[]} itemIndices
 * @param {number[][]} labelIds
 * @return {types.DeleteLabelsAction}
 */
export function deleteLabels (
  itemIndices: number[], labelIds: number[][]): types.DeleteLabelsAction {
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
