import { configureStore } from '../../src/common/configure_store'
import Session from '../../src/common/session'
import { setupSession } from '../../src/common/session_setup'
import { makeSyncMiddleware } from '../../src/common/sync_middleware'
import { Synchronizer } from '../../src/common/synchronizer'
import { DeepPartialState } from '../../src/types/state'

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
