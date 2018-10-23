import {makeItem} from './states';
import type {ItemType, ItemFunctionalType, StateType} from './types';
import {updateObject} from './util';
import {getCurrentItemViewerConfig,
  setCurrentItemViewerConfig} from './state_util';

/**
 * Create new Image item
 * @param {number} id: item id
 * @param {string} url
 * @return {ItemType}
 */
export function createItem(id: number, url: string): ItemType {
  return makeItem({id: id, index: id, url: url});
}

/**
 * Zoom image viewer by a certain ratio
 * @param {StateType} state
 * @param {number} ratio
 * @return {StateType}
 */
export function zoomImage(state: StateType, ratio: number): StateType {
  let config = getCurrentItemViewerConfig(state);
  config = updateObject(config, {viewScale: config.viewScale * ratio});
  return setCurrentItemViewerConfig(state, config);
}

/**
 * decode image item from json
 * @param {Object} json
 * @return {ItemType}
 */
export function fromJson(json: Object): ItemType {
  return makeItem({
    id: json.index,
    index: json.index,
    url: json.url,
    attributes: json.attributes,
    labels: json.labels,
  });
}

// This is necessary for different label types
export const ImageF: ItemFunctionalType = {
  createItem: createItem,
  // setActive: setActive,
};
