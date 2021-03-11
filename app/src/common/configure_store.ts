import { applyMiddleware, createStore, Middleware, Reducer } from "redux"
import { createLogger } from "redux-logger"
import thunk from "redux-thunk"
import undoable, { includeAction } from "redux-undo"

import { ADD_LABELS, DELETE_LABELS } from "../const/action"
import { makeState } from "../functional/states"
import { FullStore, ReduxState } from "../types/redux"
import { State } from "../types/state"
import { reducer } from "./reducer"

/**
 * Configure the main store for the state
 *
 * @param {Partial<State>} json: initial state
 * @param {boolean} devMode: whether to turn on dev mode
 * @param {Middleware} middleware: optional middleware for redux
 * @param initialState
 * @param devMode
 * @param middleware
 * @returns {Store<ReduxState>}
 */
export function configureStore(
  initialState: Partial<State>,
  devMode: boolean = false,
  middleware?: Middleware
): FullStore {
  const initialHistory = {
    past: Array<State>(),
    present: makeState(initialState),
    future: Array<State>()
  }

  const undoableReducer: Reducer<ReduxState> = undoable(reducer, {
    limit: 20, // Add a limit to history
    filter: includeAction([
      // Undoable actions
      ADD_LABELS,
      DELETE_LABELS
    ]),
    debug: false // Disable default debug since it misses sync actions
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
  if (middleware !== undefined) {
    allMiddleware.push(middleware)
  }

  return createStore(
    undoableReducer,
    initialHistory,
    applyMiddleware(...allMiddleware)
  )
}
