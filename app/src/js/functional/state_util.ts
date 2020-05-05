import { makeItem } from './states'
import {
  IdType,
  ItemType,
  ShapeType,
  State,
  ViewerConfigType
} from './types'

// TODO- move these to selector file and use hierarchical structure

/**
 * Get the current item from state
 * @param {State} state
 * @return {ItemType}: If no item is selected, return a new item with id -1
 */
export function getCurrentItem (state: State): ItemType {
  if (state.user.select.item < 0) {
    return makeItem()
  } else {
    return state.task.items[state.user.select.item]
  }
}

/**
 * Get shape from the state
 * @param state
 * @param itemIndex
 * @param labelId
 * @param shapeIndex
 */
export function getShape (state: State, itemIndex: number,
                          labelId: IdType, shapeIndex: number): ShapeType {
  const item = state.task.items[itemIndex]
  const shapeId = item.labels[labelId].shapes[shapeIndex]
  return item.shapes[shapeId]
}

/** Check if frame is loaded */
export function isFrameLoaded (state: State, item: number, sensor: number) {
  return state.session.itemStatuses[item].sensorDataLoaded[sensor]
}

/** Check if current frame is loaded */
export function isCurrentFrameLoaded (state: State, sensor: number) {
  return isFrameLoaded(state, state.user.select.item, sensor)
}

/**
 * Check whether item is loaded
 * @param state
 * @param item
 */
export function isItemLoaded (state: State, item: number) {
  const loadedMap =
    state.session.itemStatuses[item].sensorDataLoaded

  for (const loaded of Object.values(loadedMap)) {
    if (!loaded) {
      return false
    }
  }

  return true
}

/**
 * Check whether the current item is loaded
 * @param {State} state
 * @return boolean
 */
export function isCurrentItemLoaded (state: State): boolean {
  return isItemLoaded(state, state.user.select.item)
}

/**
 * Get config associated with viewer id
 * @param state
 * @param viewerId
 */
export function getCurrentViewerConfig (
  state: State, viewerId: number
): ViewerConfigType {
  if (viewerId in state.user.viewerConfigs) {
    return state.user.viewerConfigs[viewerId]
  }
  throw new Error(`Viewer id ${viewerId} not found`)
}
