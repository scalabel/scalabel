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

  for (let index = startIndex; index < end; index += 1) {
    trackLabels.push(_.cloneDeep(label))
    trackShapeTypes.push(shapeTypes)
    trackShapes.push(shapes)
    itemIndices.push(index)
    if (index > startIndex) {
      trackLabels[trackLabels.length - 1].manual = false
    }
  }

  return addTrack(
    itemIndices, trackLabels, trackShapeTypes, trackShapes
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
