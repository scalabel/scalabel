import _ from 'lodash'
import Session from '../common/session'
import { LabelTypeName } from '../common/types'
import { Select, State } from '../functional/types'
import { changeLabels, changeSelect, deleteLabels } from './common'
import { deleteTracks, terminateTracks } from './track'
import * as types from './types'

/**
 * Delete given label
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {DeleteLabelAction}
 */
export function deleteSelectedLabels (state: State): types.DeleteLabelsAction {
  const select = state.user.select
  const itemIndices: number[] =
    Object.keys(select.labels).map((key) => Number(key))
  const labelIds: number[][] = []
  for (const index of itemIndices) {
    labelIds.push(select.labels[index])
  }
  return deleteLabels(itemIndices, labelIds)
}

/**
 * Delete tracks corresponding to selected labels
 */
export function deleteSelectedTracks (state: State): types.DeleteLabelsAction {
  const select = state.user.select
  const tracks = []
  for (const key of Object.keys(select.labels)) {
    const index = Number(key)
    for (const labelId of select.labels[index]) {
      const label = state.task.items[index].labels[labelId]
      if (label.track in state.task.tracks) {
        tracks.push(state.task.tracks[label.track])
      }
    }
  }
  return deleteTracks(tracks)
}

/**
 * Terminate tracks corresponding to selected labels
 */
export function terminateSelectedTracks (
  state: State,
  stopIndex: number
): types.DeleteLabelsAction {
  const select = state.user.select
  const tracks = []
  for (const key of Object.keys(select.labels)) {
    const index = Number(key)
    for (const labelId of select.labels[index]) {
      const label = state.task.items[index].labels[labelId]
      if (label.track in state.task.tracks) {
        tracks.push(state.task.tracks[label.track])
      }
    }
  }
  return terminateTracks(tracks, stopIndex)
}

/**
 * Select label by ID
 * @param {number} labelId
 */
export function selectLabel (
  currentSelection: {[index: number]: number[]},
  itemIndex: number,
  labelId: number,
  category?: number,
  attributes?: {[key: number]: number[]},
  append: boolean = false
): types.ChangeSelectAction {
  return selectLabels(
    currentSelection, itemIndex, [labelId], category, attributes, append
  )
}

/**
 * Select label by ID
 * @param {number} labelId
 */
export function selectLabels (
  currentSelection: {[index: number]: number[]},
  itemIndex: number,
  labelIds: number[],
  category?: number,
  attributes?: {[key: number]: number[]},
  append: boolean = false
): types.ChangeSelectAction {
  const selectedLabels = _.cloneDeep(currentSelection)
  const newLabelIds = (append && itemIndex in selectedLabels) ?
    selectedLabels[itemIndex] : []
  for (const labelId of labelIds) {
    newLabelIds.push(labelId)
  }
  if (labelIds.length > 0 && itemIndex >= 0) {
    selectedLabels[itemIndex] = newLabelIds
  } else {
    delete selectedLabels[itemIndex]
  }

  if (!category) {
    category = 0
  }

  if (!attributes) {
    attributes = {}
  }

  return changeSelect({
    labels: (itemIndex < 0) ? {} : selectedLabels,
    category,
    attributes
  })
}

/**
 * Unselect label
 * @param currentSelection
 * @param itemIndex
 * @param labelId
 */
export function unselectLabels (
  currentSelection: {[index: number]: number[]},
  itemIndex: number,
  labelIds: number[]
) {
  const selectedLabels = _.cloneDeep(currentSelection)
  for (const labelId of labelIds) {
    const idIndex = selectedLabels[itemIndex].indexOf(labelId)
    if (idIndex >= 0) {
      selectedLabels[itemIndex].splice(idIndex, 1)
    }
  }
  return changeSelect({ labels: selectedLabels })
}

/** Change selected label and policy types */
export function selectLabel3dType (
  labelTypeName: LabelTypeName
) {
  const newSelect: Partial<Select> = {}
  const labelTypes = Session.label3dList.labelTypes
  for (let i = 0; i < labelTypes.length; i++) {
    if (labelTypes[i] === labelTypeName) {
      newSelect.labelType = i
    }
  }

  return changeSelect(newSelect)
}

/**
 * Change the properties of the label
 */
export function changeSelectedLabelsAttributes (
  state: State,
  attributes: {[key: number]: number[]}
  ): types.UpdateLabelsAction {
  const select = state.user.select
  const labelIds = Object.values(select.labels[select.item])
  const duplicatedAttributes =
    labelIds.map(((id) => ({ id, item: select.item, attributes })))
  return changeLabels(select.item, duplicatedAttributes)
}

/**
 * Change the properties of the label
 */
export function changeSelectedLabelsCategories (
  state: State,
  category: number[]
  ): types.UpdateLabelsAction {
  const select = state.user.select
  const labelIds = Object.values(select.labels[select.item])
  const duplicatedCategories =
    labelIds.map(((id) => ({ id, item: select.item, category })))
  return changeLabels(select.item, duplicatedCategories)
}
