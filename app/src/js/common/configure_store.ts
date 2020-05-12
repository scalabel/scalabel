import { AnyAction, applyMiddleware, createStore, Middleware, Reducer, Store } from 'redux'
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
export type FullStore = ReduxStore & {
  /** Thunk dispatch used for redux-thunk async actions */
  dispatch: ThunkDispatch<State, undefined, BaseAction>;
}
/**
 * Configure the main store for the state
 * @param {Partial<State>} json: initial state
 * @param {boolean} devMode: whether to turn on dev mode
 * @param {Middleware} middleware: optional middleware for redux
 * @return {Store<ReduxState>}
 */
export function configureStore (
    initialState: Partial<State>,
    debug: boolean = false,
    middleware?: Middleware): FullStore {
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
    debug
  })

  if (middleware === undefined) {
    return createStore(
      undoableReducer,
      initialHistory,
      applyMiddleware(thunk)
    )
  } else {
    return createStore(
      undoableReducer,
      initialHistory,
      applyMiddleware(thunk, middleware)
    )
  }
}
