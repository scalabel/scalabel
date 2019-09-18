import * as types from '../action/types'
import {
  getCurrentImageViewerConfig,
  setCurrentImageViewerConfig
} from './state_util'
import { makeItem } from './states'
import { ItemType, State, ViewerConfigType } from './types'
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
 * Update viewer config
 * @param {State} state
 * @param {types.UpdateImageViewerConfigAction} action
 * @return {State}
 */
export function updateImageViewerConfig (
  state: State, action: types.UpdateImageViewerConfigAction): State {
  let config: ViewerConfigType
    = getCurrentImageViewerConfig(state)
  config = updateObject(config, action.newFields)
  return setCurrentImageViewerConfig(state, config)
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
