import { configureStore } from '../../js/common/configure_store'
import Session from '../../js/common/session'
import { setupSession } from '../../js/common/session_setup'
import { DeepPartialState } from '../../js/functional/types'

/**
 * Reset the session for testing
 */
export function setupTestStore (state: DeepPartialState) {
  Session.store = configureStore({})
  setupSession(state, false)
}
