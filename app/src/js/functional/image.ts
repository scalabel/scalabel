import { makeItem } from './states'
import { ItemType } from './types'

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
