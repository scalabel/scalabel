import { ItemType, LabelType, ShapeType } from '../functional/types'
import * as types from './types'

/**
 * Create Item from url with provided creator
 * @param {Function} createItem
 * @param {string} url
 * @return {Object}
 */
export function newItem (createItem: (itemId: number, url: string) => ItemType,
                         url: string) {
  return {
    type: types.NEW_ITEM,
    createItem,
    url
  }
}

/**
 * Go to item at index
 * @param {number} index
 * @return {types.GoToItemAction}
 */
export function goToItem (index: number): types.GoToItemAction {
  return {
    type: types.GO_TO_ITEM,
    index
  }
}

/**
 * Add label to the item
 * @param {number} itemIndex
 * @param {LabelType} label
 * @param {ShapeType[]} shapes
 * @return {AddLabelAction}
 */
export function addLabel (
  itemIndex: number, label: LabelType,
  shapes: ShapeType[]): types.AddLabelAction {
  return { type: types.ADD_LABEL, itemIndex, label, shapes }
}

/**
 * Change the shape of the label
 * @param {number} itemIndex
 * @param {number} shapeId
 * @param {{}}props
 * @return {ChangeLabelShapeAction}
 */
export function changeLabelShape (
  itemIndex: number, shapeId: number, props: {}): types.ChangeLabelShapeAction {
  return { type: types.CHANGE_LABEL_SHAPE, itemIndex, shapeId, props }
}

/**
 * Change the properties of the label
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {{}}props
 * @return {ChangeLabelPropsAction}
 */
export function changeLabelProps (
  itemIndex: number, labelId: number, props: {}): types.ChangeLabelPropsAction {
  return { type: types.CHANGE_LABEL_PROPS, itemIndex, labelId, props }
}

/**
 * Delete given label
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {DeleteLabelAction}
 */
export function deleteLabel (
    itemIndex: number, labelId: number): types.DeleteLabelAction {
  return {
    type: types.DELETE_LABEL,
    itemIndex,
    labelId
  }
}

/**
 * Image tagging
 * @param {number} itemId
 * @param {number} attributeIndex
 * @param {Array<number>} selectedIndex
 * @return {Object}
 */
export function tagImage (itemId: number,
                          attributeIndex: number, selectedIndex: number[]) {
  return {
    type: types.TAG_IMAGE,
    itemId,
    attributeIndex,
    selectedIndex
  }
}

/**
 * assign Attribute to a label
 * @param {number} labelId
 * @param {Object} attributeOptions
 * @return {Object}
 */
export function changeAttribute (labelId: number, attributeOptions: object) {
  return {
    type: types.CHANGE_ATTRIBUTE,
    labelId,
    attributeOptions
  }
}

/**
 * assign Category to a label
 * @param {number} labelId
 * @param {Object} categoryOptions
 * @return {Object}
 */
export function changeCategory (labelId: number, categoryOptions: object) {
  return {
    type: types.CHANGE_CATEGORY,
    labelId,
    categoryOptions
  }
}
