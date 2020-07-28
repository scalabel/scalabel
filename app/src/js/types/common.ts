import { ActionCreator, AnyAction, Store } from 'redux'
import { ThunkAction, ThunkDispatch } from 'redux-thunk'
import { StateWithHistory } from 'redux-undo'
import { BaseAction } from '../types/action'
import { State } from '../types/functional'

/**
 * Defining the types of some general callback functions
 */
export type MaybeError = Error | null | undefined

/**
 * Types for redux state and dispatch
 */

// Types for basic undoable store
export type ReduxState = StateWithHistory<State>
export type ReduxStore = Store<ReduxState, AnyAction>

// Thunk-types extend the dispatch of the store to functions
export type ThunkDispatchType = ThunkDispatch<ReduxState, undefined, BaseAction>
export type FullStore = ReduxStore & {
  /** Thunk dispatch used for redux-thunk async actions */
  dispatch: ThunkDispatchType
}
export type ThunkActionType = ThunkAction<void, ReduxState, void, BaseAction>
export type ThunkCreatorType = ActionCreator<ThunkActionType>