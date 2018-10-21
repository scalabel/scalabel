
import type {ItemType, LabelType, StateType} from './types';
import {updateListItem, updateListItems, updateObject} from './util';

/**
 * Initialize state
 * @param {StateType} state
 * @return {StateType}
 */
export function initSession(state: StateType): StateType {
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
 * Create new label
 * @param {StateType} state: current state
 * @param {number} itemId
 * @param {Function} createLabel: label creation function
 * @param {Object} optionalAttributes
 * @return {StateType}
 */
export function newLabel(
    state: StateType,
    itemId: number,
    createLabel: (number, number, Object) => LabelType,
    optionalAttributes: Object = {}): StateType {
  let labelId = state.current.maxObjectId + 1;
  let item = updateObject(state.items[itemId],
      {labels: state.items[itemId].labels.concat([labelId])});
  let items = updateListItem(state.items, itemId, item);
  let labels = updateObject(state.labels,
      {[labelId]: createLabel(labelId, itemId, optionalAttributes)});
  let current = updateObject(state.current, {maxObjectId: labelId});
  return {
    ...state,
    items: items,
    labels: labels,
    current: current,
  };
}

/**
 * Create Item from url with provided creator
 * @param {StateType} state
 * @param {Function} createItem
 * @param {string} url
 * @return {StateType}
 */
export function newItem(
    state: StateType, createItem: (number, string) => ItemType,
    url: string): StateType {
  let id = state.items.length;
  let item = createItem(id, url);
  let items = state.items.slice();
  items.push(item);
  return {
    ...state,
    items: items,
  };
}


/**
 * Go to item at index
 * @param {StateType} state
 * @param {number} index
 * @return {StateType}
 */
export function goToItem(state: StateType, index: number): StateType {
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

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Delete given label
 * @param {StateType} state
 * @param {number} ignoredItemId
 * @param {number} ignoredLabelId
 * @return {StateType}
 */
export function deleteLabel(state: StateType, ignoredItemId: number,
                            ignoredLabelId: number): StateType {
  return state;
}

/**
 * assign Attribute to a label
 * @param {StateType} state
 * @param {number} ignoredLabelId
 * @param {object} ignoredAttributeOptions
 * @return {StateType}
 */
export function changeAttribute(state: StateType, ignoredLabelId: number,
                                ignoredAttributeOptions: Object): StateType {
  return state;
}

/**
 * change label category
 * @param {StateType} state
 * @param {number} ignoredLabelId
 * @param {object} ignoredCategoryOptions
 * @return {StateType}
 */
export function changeCategory(state: StateType, ignoredLabelId: number,
                               ignoredCategoryOptions: Object): StateType {
  return state;
}
