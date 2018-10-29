import type {ItemType, LabelType, StateType, ViewerConfigType} from './types';
import {
  removeListItems, removeObjectFields, updateListItem, updateListItems,
  updateObject,
} from './util';

/**
 * Initialize state
 * @param {StateType} state
 * @return {StateType}
 */
export function initSession(state: StateType): StateType {
  // initialize state
  let items = state.items.slice();
  for (let i = 0; i < items.length; i+=1) {
    items[i] = updateObject(items[i], {loaded: false});
  }
  state = updateObject(state, {items: items});
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
  if (index < 0 || index >= state.items.length) {
    return state;
  }
  // TODO: don't do circling when no image number is shown
  // index = (index + state.items.length) % state.items.length;
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
 * Signify a new item is loaded
 * @param {StateType} state
 * @param {number} itemIndex
 * @param {ViewerConfigType} viewerConfig
 * @return {StateType}
 */
export function loadItem(state: StateType, itemIndex: number,
                         viewerConfig: ViewerConfigType): StateType {
  return updateObject(
      state, {items: updateListItem(
          state.items, itemIndex,
            updateObject(state.items[itemIndex],
                {viewerConfig: viewerConfig, loaded: true}))});
}

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Delete given label
 * @param {StateType} state
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {StateType}
 */
export function deleteLabel(state: StateType, itemIndex: number,
                            labelId: number): StateType {
  // TODO: should we remove shapes?
  // depending on how boundary sharing is implemented.

  // remove labels
  let labels = removeObjectFields(state.labels, [labelId.toString()]);
  let items = state.items;
  if (itemIndex >= 0) {
    let item = items[itemIndex];
    items = updateListItem(items, itemIndex,
        updateObject(item,
            {labels: removeListItems(item.labels, [labelId])}));
  }

  // Reset selected object
  let current = state.current;
  if (current.label === labelId) {
    current = updateObject(current, {label: -1});
  }
  return updateObject(
      state, {current: current, labels: labels, items: items});
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
 * @param {number} labelId
 * @param {Array} categoryOptions
 * @return {StateType}
 */
export function changeCategory(state: StateType, labelId: number,
                               categoryOptions: Array<number>): StateType {
  let targetLabel = state.labels[labelId];
  let newLabel = updateObject(targetLabel, {category: categoryOptions});
  let labels = updateObject(state.labels, {[labelId]: newLabel});
  return updateObject(state, {labels: labels});
}

/**
 * Notify all the subscribers to update. it is an no-op now.
 * @param {StateType} state
 * @return {StateType}
 */
export function updateAll(state: StateType): StateType {
  return state;
}
