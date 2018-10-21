
import {makeItem} from './states';
import type {ItemType, ItemFunctionalType} from './types';

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
