import _ from "lodash"

import Session from "../common/session"
import { isValidId } from "../functional/states"
import { AddTrackAction, DeleteLabelsAction } from "../types/action"
import {
  IdType,
  INVALID_ID,
  LabelType,
  ShapeType,
  TrackType
} from "../types/state"
import { addTrack, deleteLabels } from "./common"

/**
 * Add track by duplicating label from startIndex to stopIndex
 *
 * @param label
 * @param shapeTypes
 * @param shapes
 * @param startIndex
 * @param stopIndex
 */
export function addDuplicatedTrack(
  label: LabelType,
  shapeTypes: string[],
  shapes: ShapeType[],
  startIndex?: number,
  stopIndex?: number
): AddTrackAction {
  const trackLabels: LabelType[] = []
  const trackShapeTypes: string[][] = []
  const trackShapes: ShapeType[][] = []
  const itemIndices: number[] = []

  const itemLength = Session.numItems

  if (startIndex === undefined) {
    startIndex = 0
  }

  if (stopIndex === undefined) {
    stopIndex = itemLength
  }
  const end = Math.min(stopIndex, itemLength)

  const state = Session.getState()

  let parentTrack: TrackType | undefined
  if (isValidId(label.parent)) {
    const item = state.task.items[label.item]
    const parent = item.labels[label.parent]
    if (parent.track in state.task.tracks) {
      parentTrack = state.task.tracks[parent.track]
    }
  }

  for (let index = startIndex; index < end; index += 1) {
    const cloned = _.cloneDeep(label)
    if (index > startIndex) {
      cloned.manual = false
    }

    if (parentTrack !== undefined && index in parentTrack.labels) {
      cloned.parent = parentTrack.labels[index]
    } else if (index !== cloned.item) {
      cloned.parent = INVALID_ID
    }

    trackLabels.push(cloned)
    trackShapeTypes.push(shapeTypes)
    trackShapes.push(shapes)
    itemIndices.push(index)
  }

  return addTrack(itemIndices, label.type, trackLabels, trackShapes)
}

/**
 * Delete all labels from track & track
 *
 * @param trackId
 * @param tracks
 */
export function deleteTracks(tracks: TrackType[]): DeleteLabelsAction {
  const itemLength = Session.numItems

  const itemIndices = []
  const labelIds = []

  for (let index = 0; index < itemLength; index++) {
    const toDelete: IdType[] = []
    for (const track of tracks) {
      if (index in track.labels) {
        toDelete.push(track.labels[index])
      }
    }
    itemIndices.push(index)
    labelIds.push(toDelete)
  }

  return deleteLabels(itemIndices, labelIds)
}

/**
 * Terminate track by deleting all labels in items after itemIndex
 *
 * @param track
 * @param lastIndex
 * @param tracks
 * @param firstIndexToDelete
 * @param numItems
 */
export function terminateTracks(
  tracks: TrackType[],
  firstIndexToDelete: number,
  numItems: number
): DeleteLabelsAction {
  const itemIndices = []
  const labelIds = []

  for (let index = firstIndexToDelete; index < numItems; index++) {
    const toDelete: IdType[] = []
    for (const track of tracks) {
      if (index in track.labels) {
        toDelete.push(track.labels[index])
      }
    }
    itemIndices.push(index)
    labelIds.push(toDelete)
  }

  return deleteLabels(itemIndices, labelIds)
}
