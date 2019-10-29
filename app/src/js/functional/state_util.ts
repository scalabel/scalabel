import { makeItem } from './states'
import {
  ItemType,
  ShapeType,
  State,
  ViewerConfigType
} from './types'

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
                          labelId: number, shapeIndex: number): ShapeType {
  const item = state.task.items[itemIndex]
  const shapeId = item.labels[labelId].shapes[shapeIndex]
  return item.shapes[shapeId].shape
}

/**
 * Check whether the current item is loaded
 * @param {State} state
 * @return boolean
 */
export function isItemLoaded (state: State): boolean {
  return state.session.items[state.user.select.item].loaded
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
