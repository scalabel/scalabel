import { AnyAction, applyMiddleware, createStore, Middleware, Reducer, Store } from 'redux'
import { createLogger } from 'redux-logger'
import thunk, { ThunkDispatch } from 'redux-thunk'
import undoable, { includeAction, StateWithHistory } from 'redux-undo'
import {
  ADD_LABELS,
  BaseAction,
  DELETE_LABELS
} from '../action/types'
import { makeState } from '../functional/states'
import { State } from '../functional/types'
import { reducer } from './reducer'

export type ReduxState = StateWithHistory<State>
export type ReduxStore = Store<ReduxState, AnyAction>

/**
 * Configure the main store for the state
 * @param {Partial<State>} json: initial state
 * @param {boolean} devMode: whether to turn on dev mode
 * @param {Middleware} middleware: optional middleware for redux
 * @return {Store<ReduxState>}
 */
export function configureStore (
    initialState: Partial<State>,
    devMode: boolean = false,
    middleware?: Middleware): ReduxStore & {
      /** Thunk dispatch used for redux-thunk async actions */
      dispatch: ThunkDispatch<State, undefined, BaseAction>;
    } {
  const initialHistory = {
    past: Array<State>(),
    present: makeState(initialState),
    future: Array<State>()
  }

  const undoableReducer: Reducer<ReduxState> = undoable(reducer, {
    limit: 20, // add a limit to history
    filter: includeAction([
      // undoable actions
      ADD_LABELS,
      DELETE_LABELS
    ]),
    debug: false // disable default debug since it misses sync actions
  })

  const allMiddleware: Middleware[] = [thunk]

  /**
   * If in dev mode, redux logging of normal and sync actions is enabled
   * Dev mode can be enabled by appending the query arg "?dev"
   */
  if (devMode) {
    const logger = createLogger({
      collapsed: true
    })
    allMiddleware.push(logger)
  }
  if (middleware) {
    allMiddleware.push(middleware)
  }

  return createStore(
    undoableReducer,
    initialHistory,
    applyMiddleware(...allMiddleware)
  )
}
