import {ItemType, LabelType, StateType, ViewerConfigType} from './types';
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
  const items = state.items.slice();
  for (let i = 0; i < items.length; i++) {
    items[i] = updateObject(items[i], {loaded: false});
  }
  state = updateObject(state, {items});
  if (state.current.item === -1) {
    const current = updateObject(state.current, {item: 0});
    const items = updateListItem(
        state.items, 0, updateObject(state.items[0], {active: true}));
    return updateObject(state, {current, items});
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
    createLabel: (labelId: number, itemId: number,
                  optionalAttributes: any) => LabelType,
    optionalAttributes: any = {}): StateType {
  const labelId = state.current.maxObjectId + 1;
  const item = updateObject(state.items[itemId],
      {labels: state.items[itemId].labels.concat([labelId])});
  const items = updateListItem(state.items, itemId, item);
  const labels = updateObject(state.labels,
      {[labelId]: createLabel(labelId, itemId, optionalAttributes)});
  const current = updateObject(state.current, {maxObjectId: labelId});
  return {
    ...state,
    items,
    labels,
    current
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
    state: StateType, createItem: (itemId: number, url: string) => ItemType,
    url: string): StateType {
  const id = state.items.length;
  const item = createItem(id, url);
  const items = state.items.slice();
  items.push(item);
  return {
    ...state,
    items
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
  const deactivatedItem = updateObject(state.items[state.current.item],
      {active: false});
  const activatedItem = updateObject(state.items[index], {active: true});
  const items = updateListItems(state.items,
      [state.current.item, index],
      [deactivatedItem, activatedItem]);
  const current = {...state.current, item: index};
  return updateObject(state, {items, current});
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
                {viewerConfig, loaded: true}))});
}

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Deconste given label
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
  const labels = removeObjectFields(state.labels, [labelId.toString()]);
  let items = state.items;
  if (itemIndex >= 0) {
    const item = items[itemIndex];
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
      state, {current, labels, items});
}

/**
 * assign Attribute to a label
 * @param {StateType} state
 * @param {number} ignoredLabelId
 * @param {object} ignoredAttributeOptions
 * @return {StateType}
 */
export function changeAttribute(state: StateType, ignoredLabelId: number,
                                ignoredAttributeOptions: any): StateType {
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
                               categoryOptions: number[]): StateType {
  const targetLabel = state.labels[labelId];
  const newLabel = updateObject(targetLabel, {category: categoryOptions});
  const labels = updateObject(state.labels, {[labelId]: newLabel});
  return updateObject(state, {labels});
}

/**
 * Notify all the subscribers to update. it is an no-op now.
 * @param {StateType} state
 * @return {StateType}
 */
export function updateAll(state: StateType): StateType {
  return state;
}

/**
 * turn on/off assistant view
 * @param {StateType} state
 * @return {StateType}
 */
export function toggleAssistantView(state: StateType): StateType {
  return updateObject(state, {layout:
            updateObject(state.layout, {assistantView:
                  !state.layout.assistantView})});
}
