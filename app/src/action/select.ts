import _ from "lodash"

import Session from "../common/session"
import { LabelTypeName } from "../const/common"
import * as actionTypes from "../types/action"
import { IdType, Select, State } from "../types/state"
import { changeLabelsProps, changeSelect, deleteLabels } from "./common"
import { deleteTracks, terminateTracks } from "./track"

/**
 * Delete given label
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @param state
 * @returns {DeleteLabelAction}
 */
export function deleteSelectedLabels(
  state: State
): actionTypes.DeleteLabelsAction {
  const select = state.user.select
  const itemIndices: number[] = Object.keys(select.labels).map((key) =>
    Number(key)
  )
  const labelIds: IdType[][] = []
  for (const index of itemIndices) {
    labelIds.push(select.labels[index])
  }
  return deleteLabels(itemIndices, labelIds)
}

/**
 * Delete tracks corresponding to selected labels
 *
 * @param state
 */
export function deleteSelectedTracks(
  state: State
): actionTypes.DeleteLabelsAction {
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
  return deleteTracks(_.uniq(tracks))
}

/**
 * Terminate tracks corresponding to selected labels
 *
 * @param state
 * @param stopIndex
 */
export function terminateSelectedTracks(
  state: State,
  stopIndex: number
): actionTypes.DeleteLabelsAction {
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
  return terminateTracks(_.uniq(tracks), stopIndex, state.task.items.length)
}

/**
 * Change the properties of the label
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>} props
 * @param state
 * @returns {ChangeLabelPropsAction}
 */
export function changeSelectedLabelsAttributes(
  state: State,
  attributes: { [key: number]: number[] }
): actionTypes.ChangeLabelsAction {
  const select = state.user.select
  const labelIds = Object.values(select.labels)
  let duplicatedAttributes = []
  // Tracking: propagate attributes to the end
  const selectedItem = state.task.items[select.item]
  if (selectedItem.labels[labelIds[0][0]].track !== "") {
    const labelsInTracks: { [key: number]: string[] } = {}
    for (const labelId of labelIds[0]) {
      const track = state.task.tracks[selectedItem.labels[labelId].track]
      for (const itemIndex of Object.keys(track.labels).map(Number)) {
        // Only propagate attributes to the subsequent frames
        if (itemIndex < select.item) {
          continue
        }
        if (
          itemIndex in labelsInTracks &&
          !(track.labels[itemIndex] in labelsInTracks[itemIndex])
        ) {
          labelsInTracks[itemIndex].push(track.labels[itemIndex])
        } else {
          labelsInTracks[itemIndex] = [track.labels[itemIndex]]
        }
      }
    }
    for (const value of Object.values(labelsInTracks)) {
      duplicatedAttributes.push(value.map(() => ({ attributes })))
    }
    return changeLabelsProps(
      Object.keys(labelsInTracks).map(Number),
      Object.values(labelsInTracks),
      duplicatedAttributes
    )
  }
  duplicatedAttributes = labelIds.map((arr) => arr.map(() => ({ attributes })))
  return changeLabelsProps([select.item], labelIds, duplicatedAttributes)
}

/**
 * Change the properties of the label
 *
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>} props
 * @param state
 * @param category
 * @returns {ChangeLabelPropsAction}
 */
export function changeSelectedLabelsCategories(
  state: State,
  category: number[]
): actionTypes.ChangeLabelsAction {
  const select = state.user.select
  const labelIds = Object.values(select.labels)
  let duplicatedCategories = []
  // Tracking: changes the category for the entire lifespan
  const selectedItem = state.task.items[select.item]
  if (selectedItem.labels[labelIds[0][0]].track !== "") {
    const labelsInTracks: { [key: number]: string[] } = {}
    for (const labelId of labelIds[0]) {
      const track = state.task.tracks[selectedItem.labels[labelId].track]
      for (const itemIndex of Object.keys(track.labels).map(Number)) {
        if (
          itemIndex in labelsInTracks &&
          !(track.labels[itemIndex] in labelsInTracks[itemIndex])
        ) {
          labelsInTracks[itemIndex].push(track.labels[itemIndex])
        } else {
          labelsInTracks[itemIndex] = [track.labels[itemIndex]]
        }
      }
    }
    for (const value of Object.values(labelsInTracks)) {
      duplicatedCategories.push(value.map(() => ({ category })))
    }
    return changeLabelsProps(
      Object.keys(labelsInTracks).map(Number),
      Object.values(labelsInTracks),
      duplicatedCategories
    )
  }
  duplicatedCategories = labelIds.map((arr) => arr.map(() => ({ category })))
  return changeLabelsProps([select.item], labelIds, duplicatedCategories)
}

/**
 * Select label by ID
 *
 * @param itemIndex
 * @param {number} labelId
 * @param category
 * @param append
 */
export function selectLabel(
  currentSelection: { [index: number]: IdType[] },
  itemIndex: number,
  labelId: IdType,
  category?: number,
  attributes?: { [key: number]: number[] },
  append: boolean = false
): actionTypes.ChangeSelectAction {
  return selectLabels(
    currentSelection,
    itemIndex,
    [labelId],
    category,
    attributes,
    append
  )
}

/**
 * Select label by ID
 *
 * @param currentSelection
 * @param itemIndex
 * @param labelIds
 * @param category
 * @param attributes
 * @param append
 */
export function selectLabels(
  currentSelection: { [index: number]: IdType[] },
  itemIndex: number,
  labelIds: IdType[],
  category?: number,
  attributes?: { [key: number]: number[] },
  append: boolean = false
): actionTypes.ChangeSelectAction {
  const selectedLabels = _.cloneDeep(currentSelection)
  const newLabelIds =
    append && itemIndex in selectedLabels ? selectedLabels[itemIndex] : []
  for (const labelId of labelIds) {
    newLabelIds.push(labelId)
  }
  if (labelIds.length > 0 && itemIndex >= 0) {
    selectedLabels[itemIndex] = newLabelIds
  } else {
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete selectedLabels[itemIndex]
  }

  if (category === undefined) {
    category = 0
  }

  if (attributes === undefined) {
    attributes = {}
  }

  return changeSelect({
    labels: itemIndex < 0 ? {} : selectedLabels,
    item: itemIndex,
    category,
    attributes
  })
}

/**
 * Unselect label
 *
 * @param currentSelection
 * @param itemIndex
 * @param labelId
 * @param labelIds
 */
export function unselectLabels(
  currentSelection: { [index: number]: IdType[] },
  itemIndex: number,
  labelIds: IdType[]
): actionTypes.ChangeSelectAction {
  const selectedLabels = _.cloneDeep(currentSelection)
  for (const labelId of labelIds) {
    const idIndex = selectedLabels[itemIndex].indexOf(labelId)
    if (idIndex >= 0) {
      selectedLabels[itemIndex].splice(idIndex, 1)
    }
  }
  return changeSelect({ labels: selectedLabels })
}

/**
 * Change selected label and policy types
 *
 * @param labelTypeName
 */
export function selectLabel3dType(
  labelTypeName: LabelTypeName
): actionTypes.ChangeSelectAction {
  const newSelect: Partial<Select> = {}
  const labelTypes = Session.label3dList.labelTypes
  for (let i = 0; i < labelTypes.length; i++) {
    if (labelTypes[i] === labelTypeName) {
      newSelect.labelType = i
    }
  }

  return changeSelect(newSelect)
}
