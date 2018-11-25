import * as types from './action_types';
import type {ItemType, LabelType} from '../functional/types';

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
 * @param {number} attributeIndex
 * @param {Array<number>} selectedIndex
 * @return {Object}
 */
export function tagImage(itemId: number,
                         attributeIndex: number, selectedIndex: Array<number>) {
  return {
    type: types.TAG_IMAGE,
    itemId,
    attributeIndex,
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

// Below are box2d specific
/**
 * Create a new box2d label
 * @param {number} itemId
 * @param {Object} optionalAttributes
 * @return {Object}
 */
export function newImageBox2DLabel(itemId: number, optionalAttributes: Object) {
  return {
    type: types.NEW_IMAGE_BOX2D_LABEL,
    itemId,
    optionalAttributes,
  };
}

/**
 * assign new x, y, w, h to a rectangle
 * @param {number} shapeId
 * @param {Object} targetBoxAttributes
 * @return {Object}
 */
export function changeRect(shapeId: number,
                           targetBoxAttributes: Object) {
    return {
        type: types.CHANGE_RECT, shapeId, targetBoxAttributes,
    };
}
