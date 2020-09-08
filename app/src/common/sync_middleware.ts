import { Dispatch, Middleware, MiddlewareAPI } from "redux"

import * as actionConsts from "../const/action"
import * as actionTypes from "../types/action"
import { ReduxState, ThunkDispatchType } from "../types/redux"
import { State } from "../types/state"
import { Synchronizer } from "./synchronizer"

/**
 * Handle actions that trigger a backend interaction instead of a state update
 *
 * @param action
 * @param synchronizer
 * @param state
 * @param dispatch
 */
function handleSyncAction(
  action: actionTypes.BaseAction,
  synchronizer: Synchronizer,
  state: State,
  dispatch: ThunkDispatchType
): void {
  switch (action.type) {
    case actionConsts.REGISTER_SESSION: {
      const initialState = (action as actionTypes.RegisterSessionAction)
        .initialState
      synchronizer.finishRegistration(
        initialState,
        initialState.task.config.autosave,
        initialState.session.id,
        initialState.task.config.bots,
        dispatch
      )
      break
    }
    case actionConsts.CONNECT:
      synchronizer.sendConnectionMessage(state.session.id, dispatch)
      break
    case actionConsts.DISCONNECT:
      synchronizer.handleDisconnect(dispatch)
      break
    case actionConsts.RECEIVE_BROADCAST: {
      const message = (action as actionTypes.ReceiveBroadcastAction).message
      synchronizer.handleBroadcast(message, state.session.id, dispatch)
      break
    }
    case actionConsts.SAVE:
      synchronizer.save(state.session.id, state.task.config.bots, dispatch)
      break
  }
}

/**
 * Store normal user actions for saving, either now (auto) or later (manual)
 *
 * @param action
 * @param synchronizer
 * @param state
 * @param dispatch
 */
function handleNormalAction(
  action: actionTypes.BaseAction,
  synchronizer: Synchronizer,
  state: State,
  dispatch: ThunkDispatchType
): void {
  const sessionId = state.session.id
  const autosave = state.task.config.autosave
  const bots = state.task.config.bots
  synchronizer.queueActionForSaving(action, autosave, sessionId, bots, dispatch)
}

export const makeSyncMiddleware = (
  synchronizer: Synchronizer
): Middleware<ReduxState> => {
  const syncMiddleware: Middleware<ReduxState> = ({
    dispatch,
    getState
  }: MiddlewareAPI<ThunkDispatchType, ReduxState>) => {
    return (next: Dispatch) => (action: actionTypes.BaseAction) => {
      const state = getState().present

      if (actionConsts.isSyncAction(action)) {
        // Intercept socket.io-based actions (don't call next)
        handleSyncAction(action, synchronizer, state, dispatch)
        return action
      } else {
        // Process normal actions for saving, then run them with next
        handleNormalAction(action, synchronizer, state, dispatch)
        return next(action)
      }
    }
  }
  return syncMiddleware
}
