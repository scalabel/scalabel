import _ from "lodash"

import {
  IdType,
  ItemType,
  ShapeType,
  State,
  TrackType,
  ViewerConfigType
} from "../types/state"
import { makeItem } from "./states"

// TODO- move these to selector file and use hierarchical structure

/**
 * Get the current item from state
 *
 * @param {State} state
 * @returns {ItemType}: If no item is selected, return a new item with id -1
 */
export function getCurrentItem(state: State): ItemType {
  if (state.user.select.item < 0) {
    return makeItem()
  } else {
    return state.task.items[state.user.select.item]
  }
}

/**
 * Get the number of labels on the item
 *
 * @param state
 * @param itemIndex
 */
export function getNumLabels(state: State, itemIndex: number): number {
  return _.size(state.task.items[itemIndex].labels)
}

/**
 * Get the number of shapes on the item
 *
 * @param state
 * @param itemIndex
 */
export function getNumShapes(state: State, itemIndex: number): number {
  return _.size(state.task.items[itemIndex].labels)
}

/**
 * Get the track with the specified id
 *
 * @param state
 * @param trackIndex
 * @param trackId
 */
export function getTrack(state: State, trackId: string): TrackType {
  return state.task.tracks[trackId]
}

/**
 * Get the total number of tracks
 *
 * @param state
 */
export function getNumTracks(state: State): number {
  return _.size(state.task.tracks)
}

/**
 * Get the total number of items
 *
 * @param state
 */
export function getNumItems(state: State): number {
  return _.size(state.task.items)
}

/**
 * Get the number of labels associated with the track
 *
 * @param track
 */
export function getNumLabelsForTrack(track: TrackType): number {
  return _.size(track.labels)
}

/**
 * Get the number of labels for the track with the given id
 *
 * @param state
 * @param trackId
 */
export function getNumLabelsForTrackId(state: State, trackId: IdType): number {
  const track = getTrack(state, trackId)
  return getNumLabelsForTrack(track)
}

/**
 * Get the id of the label in the track at the specified item
 *
 * @param state
 * @param trackIdx
 * @param trackId
 * @param itemIdx
 */
export function getLabelInTrack(
  state: State,
  trackId: string,
  itemIdx: number
): IdType {
  return state.task.tracks[trackId].labels[itemIdx]
}

/**
 * Get all labels that are currently selected
 *
 * @param state
 */
export function getSelectedLabels(state: State): { [index: number]: string[] } {
  return state.user.select.labels
}

/**
 * Get the category of the specified label
 *
 * @param state
 * @param itemIdx
 * @param itemIndex
 * @param labelId
 */
export function getCategory(
  state: State,
  itemIndex: number,
  labelId: IdType
): number[] {
  return state.task.items[itemIndex].labels[labelId].category
}

/**
 * Get shape from the state
 *
 * @param state
 * @param itemIndex
 * @param labelId
 * @param shapeIndex
 */
export function getShape(
  state: State,
  itemIndex: number,
  labelId: IdType,
  shapeIndex: number
): ShapeType {
  const item = state.task.items[itemIndex]
  const shapeId = item.labels[labelId].shapes[shapeIndex]
  return item.shapes[shapeId]
}

/**
 * Retrieve shapes from the state
 *
 * @param state
 * @param itemIndex
 * @param labelId
 */
export function getShapes(
  state: State,
  itemIndex: number,
  labelId: IdType
): ShapeType[] {
  const item = state.task.items[itemIndex]
  return item.labels[labelId].shapes.map((s) => item.shapes[s])
}

/**
 * Check if frame is loaded
 *
 * @param state
 * @param item
 * @param sensor
 */
export function isFrameLoaded(
  state: State,
  item: number,
  sensor: number
): boolean {
  return state.session.itemStatuses[item].sensorDataLoaded[sensor]
}

/**
 * Check if current frame is loaded
 *
 * @param state
 * @param sensor
 */
export function isCurrentFrameLoaded(state: State, sensor: number): boolean {
  return isFrameLoaded(state, state.user.select.item, sensor)
}

/**
 * Check whether item is loaded
 *
 * @param state
 * @param item
 */
export function isItemLoaded(state: State, item: number): boolean {
  const loadedMap = state.session.itemStatuses[item].sensorDataLoaded

  for (const loaded of Object.values(loadedMap)) {
    if (!loaded) {
      return false
    }
  }

  return true
}

/**
 * Check whether the current item is loaded
 *
 * @param {State} state
 * @returns boolean
 */
export function isCurrentItemLoaded(state: State): boolean {
  return isItemLoaded(state, state.user.select.item)
}

/**
 * Get config associated with viewer id
 *
 * @param state
 * @param viewerId
 */
export function getCurrentViewerConfig(
  state: State,
  viewerId: number
): ViewerConfigType {
  if (viewerId in state.user.viewerConfigs) {
    return state.user.viewerConfigs[viewerId]
  }
  throw new Error(`Viewer id ${viewerId} not found`)
}

/**
 * Get the tracks and ids of all selected tracks
 * Selected tracks can be in the same item, or different items (linking)
 *
 * @param state
 */
export function getSelectedTracks(state: State): TrackType[] {
  const selectedLabels = getSelectedLabels(state)
  const tracks: TrackType[] = []

  for (const key of Object.keys(selectedLabels)) {
    const itemIndex = Number(key)
    for (const labelId of selectedLabels[itemIndex]) {
      const trackId = state.task.items[itemIndex].labels[labelId].track
      tracks.push(getTrack(state, trackId))
    }
  }

  return tracks
}
