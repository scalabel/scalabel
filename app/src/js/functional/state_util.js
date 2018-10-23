import type {ItemType, StateType, ViewerConfigType} from './types';
import {updateListItem, updateObject} from './util';
import {makeItem} from './states';

/**
 * Get the current item from state
 * @param {StateType} state
 * @return {ItemType}: If no item is selected, return a new item with id -1
 */
export function getCurrentItem(state: StateType): ItemType {
  if (state.current.item < 0) {
    return makeItem();
  } else {
    return state.items[state.current.item];
  }
}

/**
 * Get the current item viewer config
 * @param {StateType} state
 * @return {ViewerConfigType}
 */
export function getCurrentItemViewerConfig(state: StateType): ViewerConfigType {
  return getCurrentItem(state).viewerConfig;
}

/**
 * set the current item with new item
 * @param {StateType} state
 * @param {ItemType} item
 * @return {StateType}
 */
export function setCurrentItem(state: StateType, item: ItemType): StateType {
  if (state.current.item < 0) {
    // console.error("No valid current item exists");
    return state;
  }
  return updateObject(state, {items:
        updateListItem(state.items, state.current.item, item)});
}

/**
 * Set current item viewer config
 * @param {StateType} state
 * @param {ViewerConfigType} config
 * @return {StateType}
 */
export function setCurrentItemViewerConfig(
    state: StateType, config: ViewerConfigType): StateType {
  return setCurrentItem(
      state, updateObject(getCurrentItem(state), {viewerConfig: config}));
}
