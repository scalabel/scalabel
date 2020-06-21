import { Dispatch, Middleware, MiddlewareAPI } from 'redux'
import * as types from '../action/types'
import { State } from '../functional/types'
import { Synchronizer } from './synchronizer'
import { ReduxState, ThunkDispatchType } from './types'

/**
 * Handle actions that trigger a backend interaction instead of a state update
 */
function handleSyncAction (
  action: types.BaseAction, synchronizer: Synchronizer,
  state: State, dispatch: ThunkDispatchType) {
  switch (action.type) {
    case types.REGISTER_SESSION:
      const initialState = (action as types.RegisterSessionAction).initialState
      synchronizer.finishRegistration(initialState,
        initialState.task.config.autosave,
        initialState.session.id,
        initialState.task.config.bots,
        dispatch)
      break
    case types.CONNECT:
      synchronizer.sendConnectionMessage(state.session.id, dispatch)
      break
    case types.DISCONNECT:
      synchronizer.handleDisconnect(dispatch)
      break
    case types.RECEIVE_BROADCAST:
      const message = (action as types.ReceiveBroadcastAction).message
      synchronizer.handleBroadcast(message, state.session.id, dispatch)
      break
    case types.SAVE:
      synchronizer.save(
        state.session.id, state.task.config.bots, dispatch)
      break
  }
}

/**
 * Store normal user actions for saving, either now (auto) or later (manual)
 */
function handleNormalAction (
  action: types.BaseAction, synchronizer: Synchronizer,
  state: State, dispatch: ThunkDispatchType) {
  const sessionId = state.session.id
  const autosave = state.task.config.autosave
  const bots = state.task.config.bots
  synchronizer.queueActionForSaving(action, autosave, sessionId, bots, dispatch)
}

export const makeSyncMiddleware = (synchronizer: Synchronizer) => {
  const syncMiddleware: Middleware<ReduxState> = (
    { dispatch, getState }: MiddlewareAPI<ThunkDispatchType, ReduxState>) => {

    return (next: Dispatch) => (action: types.BaseAction) => {
      const state = getState().present

      if (types.isSyncAction(action)) {
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
