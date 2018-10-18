/* @flow */

import _ from 'lodash';

import type {ItemType, LabelType, SatType} from '../../types';
import {makeSat, makeLabel} from '../../states';
import * as types from '../actions/action_types';

import {newItem as newItemF, newLabel as newLabelF} from '../sat';

import {updateObject, updateListItem, updateListItems} from '../util';

/**
 * Initialize state
 * @param {SatType} state
 * @return {SatType}
 */
function initSession(state: SatType): SatType {
  // initialize state
  if (state.current.item === -1) {
    let current = updateObject(state.current, {item: 0});
    let items = updateListItem(
        state.items, 0, updateObject(state.items[0], {active: true}));
    return updateObject(state, {current: current, items: items});
  } else {
    return state;
  }
}

/**
 * Create Item from url with provided creator
 * @param {SatType} state
 * @param {Function} createItem
 * @param {string} url
 * @return {SatType}
 */
function newItem(state: SatType, createItem: (number, string) => ItemType,
                 url: string): SatType {
  return newItemF(state, createItem, url);
}

/**
 * Go to item at index
 * @param {SatType} state
 * @param {number} index
 * @return {SatType}
 */
function goToItem(state: SatType, index: number): SatType {
  index = (index + state.items.length) % state.items.length;
  if (index === state.current.item) {
    return state;
  }
  let deactivatedItem = updateObject(state.items[state.current.item],
    {active: false});
  let activatedItem = updateObject(state.items[index], {active: true});
  let items = updateListItems(state.items,
    [state.current.item, index],
    [deactivatedItem, activatedItem]);
  let current = {...state.current, item: index};
  return updateObject(state, {items: items, current: current});
}

/**
 * Create new label from given creator function
 * @param {SatType} state
 * @param {number} itemId
 * @param {Function} createLabel
 * @param {Object} optionalAttributes
 * @return {SatType}
 */
function newLabel(state: SatType,
                  itemId: number,
                  createLabel: (number, number, Object) => LabelType,
                  optionalAttributes: Object = {}): SatType {
  return newLabelF(state, itemId, createLabel, optionalAttributes);
}

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Delete given label
 * @param {SatType} state
 * @param {number} ignoredItemId
 * @param {number} ignoredLabelId
 * @return {SatType}
 */
function deleteLabel(
    state: SatType, ignoredItemId: number, ignoredLabelId: number): SatType {
  return state;
}

/**
 * Create a Tag label
 * @param {number} labelId
 * @param {number} itemId
 * @param {Object} optionalAttributes
 * @return {LabelType}
 */
function createTagLabel(labelId: number, itemId: number,
                        optionalAttributes: Object): LabelType {
  return makeLabel({id: labelId, item: itemId, attributes: optionalAttributes});
}

/**
 *Image tagging
 * @param {SatType} state
 * @param {number} itemId
 * @param {number} attributeName
 * @param {number} attributeValue
 * @return {SatType}
 */
function tagImage(state: SatType, itemId: number, attributeName: string,
                 attributeValue: Array<number>): SatType {
  let attributes = {[attributeName]: attributeValue};
  let item = state.items[itemId];
  if (item.labels.length > 0) {
    let labelId = item.labels[0];
    // be careful about this merge
    return _.merge({}, state, {labels: {[labelId]: {attributes: attributes}}});
  }
  return newLabel(state, itemId, createTagLabel, attributes);
}

/**
 * assign Attribute to a label
 * @param {SatType} state
 * @param {number} ignoredLabelId
 * @param {object} ignoredAttributeOptions
 * @return {SatType}
 */
function changeAttribute(state: SatType, ignoredLabelId: number,
                         ignoredAttributeOptions: Object): SatType {
  return state;
}

/**
 * change label category
 * @param {SatType} state
 * @param {number} ignoredLabelId
 * @param {object} ignoredCategoryOptions
 * @return {SatType}
 */
function changeCategory(state: SatType, ignoredLabelId: number,
                        ignoredCategoryOptions: Object): SatType {
  return state;
}

/**
 * Reducer
 * @param {SatType} currState
 * @param {object} action
 * @return {SatType}
 */
export default function(currState: SatType = makeSat(),
                        action: Object): SatType {
  // Appending actions to action array
  let newActions = currState.actions.slice();
  newActions.push(action);
  let state = {...currState, actions: newActions};
  // Apply reducers to state
  switch (action.type) {
    case types.INIT_SESSION:
      return initSession(state);
    case types.NEW_ITEM:
      return newItem(state, action.createItem, action.url);
    case types.GO_TO_ITEM:
      return goToItem(state, action.index);
    case types.NEW_LABEL:
      return newLabel(state, action.itemId,
        action.createLabel, action.optionalAttributes);
    case types.DELETE_LABEL:
      return deleteLabel(state, action.itemId, action.labelId);
    case types.TAG_IMAGE:
      return tagImage(state, action.itemId,
        action.attributeName, action.selectedIndex);
    case types.CHANGE_ATTRIBUTE:
      return changeAttribute(state, action.labelId, action.attributeOptions);
    case types.CHANGE_CATEGORY:
      return changeCategory(state, action.labelId, action.categoryOptions);
    // case types.NEW_PROJECT: // TODO: need this?
    //   return setInitialState(state, action.options);
    default:
  }
  return state;
}
