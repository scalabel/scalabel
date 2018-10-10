/* @flow */

import _ from 'lodash';

import type {ItemType, LabelType, SatType} from '../../types';
import {makeSat} from '../../states';
import * as types from '../actions/action_types';

import {newItem as newItemF, newLabel as newLabelF} from '../sat';

import {updateObject, updateListItems} from '../util';

/**
 * Create Item from url with provided creator
 * @param {SatType} state
 * @param {Function} createItem
 * @param {string} url
 * @return {SatType}
 */
function newItem(state: SatType, createItem: (number, string) => ItemType,
                 url: string) {
  return newItemF(state, createItem, url);
}

/**
 * Go to item at index
 * @param {SatType} state
 * @param {number} index
 * @return {SatType}
 */
function goToItem(state: SatType, index: number) {
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
 * @param {Function} createLabel
 * @param {Object} optionalAttributes
 * @return {SatType}
 */
function newLabel(state: SatType, createLabel: (number, Object) => LabelType,
                  optionalAttributes: Object = {}) {
  let newState = newLabelF(state, createLabel, optionalAttributes);
  return newState;
}

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Delete given label
 * @param {SatType} state
 * @param {number} itemId
 * @param {number} labelId
 * @return {SatType}
 */
function deleteLabel(state: SatType, itemId: number, labelId: number) { // eslint-disable-line
  return state;
}

/**
 *Image tagging
 * @param {SatType} state
 * @param {number} itemId
 * @param {number} attributeName
 * @param {number} selectedIndex
 * @return {SatType}
 */
 function tagImage(state: SatType, itemId: number, attributeName: string,
                   selectedIndex: number) {
   let attributes = updateObject(state.items[itemId].attributes,
     {[attributeName]: selectedIndex});
   // be careful about this merge
   return _.merge({}, state, {items: {[itemId]: {attributes: attributes}}});
 }

/**
 * assign Attribute to a label
 * @param {SatType} state
 * @param {number} labelId
 * @param {object} attributeOptions
 * @return {SatType}
 */
function changeAttribute(state: SatType,
                         labelId: number, attributeOptions: Object) { // eslint-disable-line
  return state;
}

/**
 * change label category
 * @param {SatType} state
 * @param {number} labelId
 * @param {object} categoryOptions
 * @return {SatType}
 */
function changeCategory(state: SatType,
                        labelId: number, categoryOptions: Object) { // eslint-disable-line
  return state;
}

/**
 * Reducer
 * @param {SatType} currState
 * @param {object} action
 * @return {SatType}
 */
export default function(currState: SatType = makeSat(), action: Object) {
  // Appending actions to action array
  let newActions = currState.actions.slice();
  newActions.push(action);
  let state = {...currState, actions: newActions};
  // Apply reducers to state
  switch (action.type) {
    case types.NEW_ITEM:
      return newItem(state, action.createItem, action.url);
    case types.GO_TO_ITEM:
      return goToItem(state, action.index);
    case types.NEW_LABEL:
      return newLabel(state, action.createLabel, action.optionalAttributes);
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
