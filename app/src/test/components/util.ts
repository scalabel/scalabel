import { configureStore } from '../../js/common/configure_store'
import Session from '../../js/common/session'
import { setupSession } from '../../js/common/session_setup'
import { makeSyncMiddleware } from '../../js/common/sync_middleware'
import { Synchronizer } from '../../js/common/synchronizer'
import { DeepPartialState } from '../../js/types/state'

/**
 * Reset the session for testing
 */
export function setupTestStore (state: DeepPartialState) {
  Session.store = configureStore({})
  setupSession(state, '', false)
}

/**
 * Reset the session for testing, and add sync middleware
 */
export function setupTestStoreWithMiddleware (
  state: DeepPartialState, synchronizer: Synchronizer) {
  const syncMiddleware = makeSyncMiddleware(synchronizer)

  Session.store = configureStore({}, false, syncMiddleware)
  setupSession(state, '', false)
}
