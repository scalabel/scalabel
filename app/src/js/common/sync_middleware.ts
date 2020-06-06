import { Dispatch, Middleware, MiddlewareAPI } from 'redux'
import * as types from '../action/types'
import { State } from '../functional/types'
import { ReduxState } from './configure_store'
import { Synchronizer } from './synchronizer'

/**
 * Handles actions related to socket.io and synchronization
 */
function handleSyncAction  (
  action: types.BaseAction, synchronizer: Synchronizer, state: State) {
  switch (action.type) {
    case types.REGISTER_SESSION:
      const initialState = (action as types.RegisterSessionAction).initialState
      synchronizer.finishRegistration(initialState,
        initialState.task.config.autosave,
        initialState.session.id,
        initialState.task.config.bots)
      break
    case types.CONNECT:
      synchronizer.sendConnectionMessage(state.session.id)
      break
    case types.DISCONNECT:
      synchronizer.handleDisconnect()
      break
    case types.RECEIVE_BROADCAST:
      const message = (action as types.ReceiveBroadcastAction).message
      synchronizer.handleBroadcast(message)
      break
    case types.SAVE:
      synchronizer.sendQueuedActions(
        state.session.id, state.task.config.bots)
      break
  }
}

/**
 * Handles autosaving of a normal action
 */
function handleNormalAction (
  action: types.BaseAction, synchronizer: Synchronizer, state: State) {
  const sessionId = state.session.id
  const autosave = state.task.config.autosave
  const bots = state.task.config.bots
  synchronizer.logAction(action, autosave, sessionId, bots)
}

export const makeSyncMiddleware = (synchronizer: Synchronizer) => {
  const syncMiddleware: Middleware<ReduxState> = (
    { getState }: MiddlewareAPI<Dispatch, ReduxState>) => {

    return (next: Dispatch) => (action: types.BaseAction) => {
      const state = getState().present

      if (types.isSyncAction(action)) {
        // Handle socket events
        handleSyncAction(action, synchronizer, state)
        return action
      } else {
        handleNormalAction(action, synchronizer, state)
        return next(action)
      }
    }
  }
  return syncMiddleware
}
