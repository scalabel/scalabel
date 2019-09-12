import _ from 'lodash'
import * as fp from 'lodash/fp'

// TODO: need a deep merge? i.e. to update below:
// state.items[itemIndex].attributes[attributeId].selectedIndex
// something like updateIn() in immutable

/**
 * This should be used to update the immutable objects
 * @param {{}} object
 * @param {{}} newFields
 * @return {{}}
 */
export function updateObject<T> (object: T, newFields: Partial<T>): T {
  return { ...object, ...newFields }
}

/**
 * This should be used to update the immutable array
 * @param {Array} array
 * @param {number} index
 * @param {T} item
 * @return {Array}
 */
export function updateListItem<T> (
    array: T[], index: number, item: T): T[] {
  array = array.slice()
  array[index] = item
  return array
}

/**
 * This should be used to update the immutable array
 * with multiple items, less .slice() call needed
 * @param {Array} array
 * @param {Array} indices
 * @param {Array} items
 * @return {Array}
 */
export function updateListItems<T> (
  array: T[], indices: number[], items: T[]): T[] {
  array = array.slice()
  for (let i = 0; i < indices.length; i++) {
    array[indices[i]] = items[i]
  }
  return array
}

/**
 * Add an item to an array
 * @param {Array} items
 * @param {T} item
 * @return {Array}
 */
export function addListItem<T> (items: T[], item: T): T[] {
  return items.concat([item])
}

/**
 * Get numerical keys from an object
 * @param {{[key: number]: T}} object
 */
export function getObjectKeys<T> (object: {[key: number]: T}): number[] {
  const keys: number[] = []
  _.forEach(object, (_value, k) => {
    keys.push(Number(k))
  })
  return keys
}

/**
 * Remove fields from an object
 * @param {T} target
 * @param {Array<keyof T>} fields
 * @return {T}
 */
export function removeObjectFields<T> (
    target: T, fields: Array<keyof T>): T {
  target = { ...target }
  for (const f of fields) {
    delete target[f]
  }
  return target
}

interface IdSingle {
  /** ID */
  id: number
}

/**
 * remove list items by item id
 * @param {T[]} items
 * @param {number[]} ids
 * @return {T[]}
 */
export function removeListItemsById (items: IdSingle[],
                                     ids: number[]): IdSingle[] {
  return fp.remove(
      (item: IdSingle) => fp.indexOf(item.id, ids) >= 0)(items)
}

/**
 * Remove list items by equivalence
 * @param {T[]} items
 * @param {T[]} a
 * @return {T[]}
 */
export function removeListItems<T> (items: T[], a: T[]): T[] {
  return fp.remove((item) => fp.indexOf(item, a) >= 0, items)
}

/**
 * Pick values of keys from an object
 * @param {{[K]: T}} object
 * @param {number[]} keys
 */
export function pickObject<T> (
  object: { [key: number]: T }, keys: number[]): {[key: number]: T} {
  const newObject: { [key: number]: T } = {}
  keys.forEach((key) => {
    newObject[key] = object[key]
  })
  return newObject
}

/**
 * pick array elements based on indices
 * @param {T[]} array
 * @param {number[]} indices
 */
export function pickArray<T> (array: T[], indices: number[]): T[] {
  const newArray: T[] = []
  indices.forEach((index) => {
    newArray.push(array[index])
  })
  return newArray
}

/**
 * Assign elements to an array by indices
 * @param {T[]} array
 * @param {T[]} newElements
 * @param {number[]} indices
 */
export function assignToArray<T> (
    array: T[], newElements: T[], indices: number[]): T[] {
  // This function should not mutate the input array
  array = [...array]
  indices.forEach((index, i) => {
    array[index] = newElements[i]
  })
  return array
}
