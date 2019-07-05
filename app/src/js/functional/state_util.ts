import { makeItem } from './states'
import { ItemType, State, ViewerConfigType } from './types'
import { updateListItem, updateObject } from './util'

/**
 * Get the current item from state
 * @param {State} state
 * @return {ItemType}: If no item is selected, return a new item with id -1
 */
export function getCurrentItem (state: State): ItemType {
  if (state.current.item < 0) {
    return makeItem()
  } else {
    return state.items[state.current.item]
  }
}

/**
 * Get the current item viewer config
 * @param {State} state
 * @return {ViewerConfigType}
 */
export function getCurrentItemViewerConfig (
    state: State): ViewerConfigType {
  return getCurrentItem(state).viewerConfig
}

/**
 * set the current item with new item
 * @param {State} state
 * @param {ItemType} item
 * @return {State}
 */
export function setCurrentItem (state: State, item: ItemType): State {
  if (state.current.item < 0) {
    // console.error("No valid current item exists");
    return state
  }
  return updateObject(state, {items:
        updateListItem(state.items, state.current.item, item)})
}

/**
 * Set current item viewer config
 * @param {State} state
 * @param {ViewerConfigType} config
 * @return {State}
 */
export function setCurrentItemViewerConfig (
    state: State, config: ViewerConfigType): State {
  return setCurrentItem(
      state, updateObject(getCurrentItem(state), { viewerConfig: config }))
}
