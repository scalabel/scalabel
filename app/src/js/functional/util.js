/* @flow */

/**
 * This should be used to update the immutable objects
 * @param {{}} object
 * @param {{}} newFields
 * @return {{}}
 */
export function updateObject(object: {}, newFields: {}): {} {
  return {...object, ...newFields};
}

/**
 * This should be used to update the immutable array
 * @param {Array} array
 * @param {number} index
 * @param {any} item
 * @return {Array}
 */
export function updateListItem<T>(
    array: Array<T>, index: number, item: T): Array<T> {
  array = array.slice();
  array[index] = item;
  return array;
}

/**
 * Add an item to an array
 * @param {Array} items
 * @param {any} item
 * @return {Array}
 */
export function addListItem<T>(items: Array<T>, item: T): Array<T> {
  return items.concat([item]);
}
