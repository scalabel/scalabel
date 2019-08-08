import * as types from '../action/types'
import {getCurrentItemViewerConfig,
  setCurrentItemViewerConfig} from './state_util'
import { makeItem } from './states'
import { ImageViewerConfigType, ItemFuncType, ItemType, State, ViewerConfigType } from './types'
import { updateObject } from './util'

/**
 * Create new Image item
 * @param {number} id: item id
 * @param {string} url
 * @return {ItemType}
 */
export function createItem (id: number, url: string): ItemType {
  return makeItem({ id, index: id, url })
}

/**
 * Zoom image viewer by a certain ratio
 * @param {State} state
 * @param {types.ImageZoomAction} action
 * @return {State}
 */
export function zoomImage (
    state: State, action: types.ImageZoomAction): State {
  const [ratio, offsetX, offsetY] = [action.ratio,
    action.viewOffsetX, action.viewOffsetY]
  let config: ViewerConfigType
        = getCurrentItemViewerConfig(state) as ImageViewerConfigType
  config = updateObject(config, {
    viewScale: config.viewScale * ratio,
    viewOffsetX: offsetX,
    viewOffsetY: offsetY})
  return setCurrentItemViewerConfig(state, config)
}

/**
 * decode image item from json
 * @param {ItemType} json
 * @return {ItemType}
 */
// TODO: check correctness...
export function fromJson (json: ItemType): ItemType {
  return makeItem({
    id: json.index,
    index: json.index,
    url: json.url,
    // attributes: json.attributes,
    labels: json.labels
  })
}

// This is necessary for different label types
export type ImageF = ItemFuncType
