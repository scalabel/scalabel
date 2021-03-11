import _ from "lodash"

import { IdType, LabelIdMap, State } from "../../src/types/state"

/**
 * Find the new label that is not already in the labelIds
 *
 * @param labels
 * @param labelIds
 */
export function findNewLabels(
  labels: LabelIdMap,
  labelIds: IdType[]
): IdType[] {
  return _.filter(_.keys(labels), (id) => !labelIds.includes(id))
}

/**
 * Find the new label that is not already in the labelIds
 *
 * @param labels
 * @param state
 * @param itemIndex
 * @param labelIds
 */
export function findNewLabelsFromState(
  state: State,
  itemIndex: number,
  labelIds: IdType[]
): IdType[] {
  const labels = state.task.items[itemIndex].labels
  return _.filter(_.keys(labels), (id) => !labelIds.includes(id))
}

/**
 * Find the new label that is not already in the labelIds
 *
 * @param state
 * @param trackIds
 */
export function findNewTracksFromState(
  state: State,
  trackIds: IdType[]
): IdType[] {
  const tracks = state.task.tracks
  return _.filter(_.keys(tracks), (id) => !trackIds.includes(id))
}
