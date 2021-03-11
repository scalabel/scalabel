import { ActionCreator, AnyAction, Store } from "redux"
import { ThunkAction, ThunkDispatch } from "redux-thunk"
import { StateWithHistory } from "redux-undo"

import { BaseAction } from "./action"
import { State } from "./state"

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
// export interface ThunkActionType
//   extends ThunkAction<void, ReduxState, void, BaseAction> {
//   type: string
// }
export type ThunkCreatorType = ActionCreator<ThunkActionType>
