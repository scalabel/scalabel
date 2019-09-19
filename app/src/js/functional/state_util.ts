import { makeItem } from './states'
import {
  ImageViewerConfigType,
  ItemType,
  PointCloudViewerConfigType,
  ShapeType,
  State,
  UserType,
  ViewerConfigType
} from './types'
import { updateObject } from './util'

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
 * Get the current item viewer config
 * @param {State} state
 * @return {ViewerConfigType}
 */
export function getCurrentImageViewerConfig (state: State):
  ImageViewerConfigType {
  return state.user.imageViewerConfig
}

/**
 * Get the current item viewer config
 * @param {State} state
 * @return {ViewerConfigType}
 */
export function getCurrentPointCloudViewerConfig (state: State):
  PointCloudViewerConfigType {
  return state.user.pointCloudViewerConfig
}

/**
 * Set current image viewer config
 * @param {State} state
 * @param {ViewerConfigType} config
 * @return {State}
 */
export function setCurrentImageViewerConfig (
    state: State, config: ViewerConfigType): State {
  const newUser: UserType = updateObject(state.user, {
    imageViewerConfig: config as ImageViewerConfigType
  })
  return updateObject(state, { user: newUser })
}

/**
 * Set current point cloud viewer config
 * @param {State} state
 * @param {ViewerConfigType} config
 * @return {State}
 */
export function setCurrentPointCloudViewerConfig (
    state: State, config: ViewerConfigType): State {
  const newUser: UserType = updateObject(state.user, {
    pointCloudViewerConfig: config as PointCloudViewerConfigType
  })
  return updateObject(state, { user: newUser })
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
