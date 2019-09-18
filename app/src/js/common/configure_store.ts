import { createStore, Reducer, Store } from 'redux'
import undoable, { includeAction, StateWithHistory } from 'redux-undo'
import {
  ADD_LABELS,
  DELETE_LABELS
} from '../action/types'
import { makeState } from '../functional/states'
import { State } from '../functional/types'
import { reducer } from './reducer'

/**
 * Configure the main store for the state
 * @param {Partial<State>} json: initial state
 * @param {boolean} devMode: whether to turn on dev mode
 * @return {Store<StateWithHistory<State>>}
 */
export function configureStore (
    initialState: Partial<State>,
    devMode: boolean = false): Store<StateWithHistory<State>> {
  const initialHistory = {
    past: Array<State>(),
    present: makeState(initialState),
    future: Array<State>()
  }

  const undoableReduer: Reducer<StateWithHistory<State>> = undoable(reducer, {
    limit: 20, // add a limit to history
    filter: includeAction([
      // undoable actions
      ADD_LABELS,
      DELETE_LABELS
    ]),
    debug: devMode
  })

  return createStore(undoableReduer, initialHistory)
}
