import {makeItem} from './states';
import {ItemType, ItemFunctionalType, StateType, ImageViewerConfigType} from './types';
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
  return makeItem({id, index: id, url});
}

/**
 * Zoom image viewer by a certain ratio
 * @param {StateType} state
 * @param {number} ratio
 * @return {StateType}
 */
export function zoomImage(state: StateType, ratio: number): StateType {
  let config = getCurrentItemViewerConfig(state);
  if (config) {
    config = updateObject(config,
        {viewScale: (config as ImageViewerConfigType).viewScale * ratio});
    return setCurrentItemViewerConfig(state, config);
  } else {
    return state;
  }
}

/**
 * decode image item from json
 * @param {Object} json
 * @return {ItemType}
 */
export function fromJson(json: any): ItemType {
  return makeItem({
    id: json.index,
    index: json.index,
    url: json.url,
    attributes: json.attributes,
    labels: json.labels
  });
}

// This is necessary for different label types
export type ImageF = ItemFunctionalType;
