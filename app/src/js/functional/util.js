import * as fp from 'lodash/fp';

// TODO: need a deep merge? i.e. to update below:
// state.items[itemId].attributes[attributeId].selectedIndex
// something like updateIn() in immutable

/**
 * This should be used to update the immutable objects
 * @param {{}} object
 * @param {{}} newFields
 * @return {{}}
 */
export function updateObject(object: Object, newFields: Object): Object {
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
 * This should be used to update the immutable array
 * with multiple items, less .slice() call needed
 * @param {Array} array
 * @param {Array} indices
 * @param {Array} items
 * @return {Array}
 */
export function updateListItems<T>(
  array: Array<T>, indices: Array<number>, items: Array<T>): Array<T> {
  array = array.slice();
  for (let i=0; i < indices.length; i++) {
    array[indices[i]] = items[i];
  }
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

/**
 * Remove fields from an object
 * @param {T} object
 * @param {Array<any>} fields
 * @return {T}
 */
export function removeObjectFields<T>(object: T, fields: Array<any>): T {
  object = {...object};
  for (let f of fields) {
    delete object[f];
  }
  return object;
}

/**
 * remove list items by item id
 * @param {Array<T>} items
 * @param {Array<number>} ids
 * @return {Array<T>}
 */
export function removeListItemsById<T: {id: number}>(
    items: Array<T>, ids: Array<number>): Array<T> {
  return fp.remove((item: T) => fp.indexOf(item.id, ids) >= 0)(items);
}

/**
 * Remove list items by equivalence
 * @param {Array<T>} items
 * @param {Array<T>} a
 * @return {Array<T>}
 */
export function removeListItems<T>(items: Array<T>, a: Array<T>): Array<T> {
  return fp.remove((item) => fp.indexOf(item, a) >= 0, items);
}
