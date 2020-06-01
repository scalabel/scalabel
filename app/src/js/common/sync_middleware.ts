import { Dispatch, Middleware, MiddlewareAPI } from 'redux'
import { setStatusAfterConnect } from '../action/common'
import * as types from '../action/types'
import { ReduxState } from './configure_store'
import { Synchronizer } from './synchronizer'

export const makeSyncMiddleware = (synchronizer: Synchronizer) => {
  const syncMiddleware: Middleware<ReduxState> = (
    { dispatch, getState }: MiddlewareAPI<Dispatch, ReduxState>) => {

    return (next: Dispatch) => (action: types.BaseAction) => {
      const state = getState().present
      const sessionId = state.session.id
      const autosave = state.task.config.autosave

      // Handle socket events
      if (types.isSyncAction(action)) {
        switch (action.type) {
          case types.REGISTER_SESSION:
          case types.CONNECT:
            synchronizer.sendConnectionMessage(sessionId)
            dispatch(setStatusAfterConnect())
            break
          case types.DISCONNECT:
          case types.RECEIVE_BROADCAST:
          default:
            // throw error- should all be covered
        }
        return action
      } else {
        if (sessionId === action.sessionId && !action.frontendOnly &&
          !types.isSessionAction(action)) {
            // sync.queueAction
        }

        return next(action)
      }
    }
  }
  return syncMiddleware
}
