import { createStore } from 'redux'
import undoable, { includeAction } from 'redux-undo'
import { makeState } from '../functional/states'
import { reducer } from './reducer'

import {
// SAT specific actions
  ADD_LABEL,
  CHANGE_ATTRIBUTE,
  CHANGE_CATEGORY,
  DELETE_LABEL,
  GO_TO_ITEM,
  IMAGE_ZOOM,
  NEW_ITEM,
  TAG_IMAGE,
  TOGGLE_ASSISTANT_VIEW
} from '../action/types'

/**
 * Configure the main store for the state
 * @param {Object} json: initial state
 * @param {boolean} devMode: whether to turn on dev mode
 * @return {Object}
 */
export function configureStore (
    json: any = {}, devMode: boolean = false): any {
  let store
  const initialHistory = {
    past: [],
    present: makeState(json),
    future: []
  }

  store = createStore(undoable(reducer, {
    limit: 20, // add a limit to history
    filter: includeAction([
      // undoable actions
      NEW_ITEM,
      GO_TO_ITEM,
      IMAGE_ZOOM,
      ADD_LABEL,
      DELETE_LABEL,
      TAG_IMAGE,
      CHANGE_ATTRIBUTE,
      CHANGE_CATEGORY,
      TOGGLE_ASSISTANT_VIEW
    ]),
    debug: devMode
  }), initialHistory as any)

  return store
}

/**
 * Create fast and partial store for interactive mode
 * @return {Object}
 */
export function configureFastStore (): any {
  return createStore(reducer, makeState())
}
