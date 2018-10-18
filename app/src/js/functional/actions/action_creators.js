/* @flow */
import * as types from './action_types';
import type {ItemType, LabelType} from '../../types';

/**
 * Create Item from url with provided creator
 * @param {Function} createItem
 * @param {string} url
 * @return {Object}
 */
export function newItem(createItem: (number, string) => ItemType,
                        url: string) {
  return {
    type: types.NEW_ITEM,
    createItem,
    url,
  };
}

/**
 * Go to item at index
 * @param {number} index
 * @return {Object}
 */
export function goToItem(index: number) {
  return {
    type: types.GO_TO_ITEM,
    index,
  };
}

/**
 * Create new label from given creator function
 * @param {number} itemId
 * @param {Function} createLabel
 * @param {Object} optionalAttributes
 * @return {Object}
 */
export function newLabel(itemId: number,
                         createLabel: (number, number, Object) => LabelType,
                         optionalAttributes: Object = Object) {
  return {
    type: types.NEW_LABEL,
    itemId,
    createLabel,
    optionalAttributes,
  };
}

/**
 * Delete given label
 * @param {number} itemId
 * @param {number} labelId
 * @return {Object}
 */
export function deleteLabel(itemId: number, labelId: number) {
  return {
    type: types.DELETE_LABEL,
    itemId,
    labelId,
  };
}

/**
 * Image tagging
 * @param {number} itemId
 * @param {number} attributeName
 * @param {Array<number>} selectedIndex
 * @return {Object}
 */
export function tagImage(itemId: number,
                         attributeName: string, selectedIndex: Array<number>) {
  return {
    type: types.TAG_IMAGE,
    itemId,
    attributeName,
    selectedIndex,
  };
}

/**
 * assign Attribute to a label
 * @param {number} labelId
 * @param {Object} attributeOptions
 * @return {Object}
 */
export function changeAttribute(labelId: number, attributeOptions: Object) {
  return {
    type: types.CHANGE_ATTRIBUTE,
    labelId,
    attributeOptions,
  };
}

/**
 * assign Category to a label
 * @param {number} labelId
 * @param {Object} categoryOptions
 * @return {Object}
 */
export function changeCategory(labelId: number, categoryOptions: Object) {
  return {
    type: types.CHANGE_CATEGORY,
    labelId,
    categoryOptions,
  };
}
