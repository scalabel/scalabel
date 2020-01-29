import _ from 'lodash'
import Session from '../common/session'
import { TrackType } from '../functional/types'
import { DELETE_LABELS, DeleteLabelsAction } from './types'

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
