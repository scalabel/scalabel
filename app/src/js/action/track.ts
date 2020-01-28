import _ from 'lodash'
import Session from '../common/session'
import { LabelType, ShapeType, TrackType } from '../functional/types'
import { addTrack } from './common'
import { AddTrackAction, DELETE_LABELS, DeleteLabelsAction } from './types'

/**
 * Add track by duplicating label from startIndex to stopIndex
 * @param label
 * @param shapeTypes
 * @param shapes
 * @param startIndex
 * @param stopIndex
 */
export function addDuplicatedTrack (
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

  if (!startIndex) {
    startIndex = 0
  }

  if (!stopIndex) {
    stopIndex = itemLength
  }
  const end = Math.min(stopIndex, itemLength)

  const state = Session.getState()

  let parentTrack: TrackType | undefined
  if (label.parent >= 0) {
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

    if (parentTrack && index in parentTrack.labels) {
      cloned.parent = parentTrack.labels[index]
    } else if (index !== cloned.item) {
      cloned.parent = -1
    }

    trackLabels.push(cloned)
    trackShapeTypes.push(shapeTypes)
    trackShapes.push(shapes)
    itemIndices.push(index)
  }

  return addTrack(
    itemIndices, label.type, trackLabels, trackShapeTypes, trackShapes
  )
}

/**
 * Delete all labels from track & track
 * @param trackId
 */
export function deleteTracks (
  tracks: TrackType[]
): DeleteLabelsAction {
  const itemLength = Session.numItems

  const itemIndices = []
  const labelIds = []

  for (let index = 0; index < itemLength; index++) {
    const toDelete: number[] = []
    for (const track of tracks) {
      if (index in track.labels) {
        toDelete.push(track.labels[index])
      }
    }
    itemIndices.push(index)
    labelIds.push(toDelete)
  }

  return {
    type: DELETE_LABELS,
    sessionId: Session.id,
    itemIndices,
    labelIds
  }
}

/**
 * Terminate track by deleting all labels in items after itemIndex
 * @param track
 * @param lastIndex
 */
export function terminateTracks (
  tracks: TrackType[],
  lastIndex: number
): DeleteLabelsAction {
  const itemLength = Session.numItems

  const itemIndices = []
  const labelIds = []

  for (let index = lastIndex + 1; index < itemLength; index++) {
    const toDelete: number[] = []
    for (const track of tracks) {
      if (index in track.labels) {
        toDelete.push(track.labels[index])
      }
    }
    itemIndices.push(index)
    labelIds.push(toDelete)
  }

  return {
    type: DELETE_LABELS,
    sessionId: Session.id,
    itemIndices,
    labelIds
  }
}
