import { createSelector } from 'reselect'
import { sprintf } from 'sprintf-js'
import { ConfigType, ConnectionStatus, State } from '../functional/types'

/**
 * Load the task config
 */
export function getConfig (state: State): ConfigType {
  return state.task.config
}

/**
 * Get link to the dashboard
 */
export const getDashboardLink = createSelector(
  [ getConfig ],
  (config: ConfigType) => {
    return sprintf('/vendor?project_name=%s', config.projectName)
  }
)

/**
 * Get the status of the session
 */
export function getSessionStatus (state: State): ConnectionStatus {
  return state.session.status
}

/**
 * Get display text based on the session status
 */
export const getStatusText = createSelector(
  [ getSessionStatus ],
  (status: ConnectionStatus) => {
    switch (status) {
      case ConnectionStatus.SAVING: {
        return 'Saving in progress...'
      }
      case ConnectionStatus.RECONNECTING: {
        return 'Trying to reconnect...'
      }
      case ConnectionStatus.COMPUTING: {
        return 'Model predictions in progress..'
      }
      case ConnectionStatus.COMPUTE_DONE:
      case ConnectionStatus.NOTIFY_COMPUTE_DONE: {
        return 'Model predictions complete.'
      }
      case ConnectionStatus.SAVED:
      case ConnectionStatus.NOTIFY_SAVED: {
        return 'All progress saved.'
      }
      default: {
        return 'All progress saved.'
      }
    }
  }
)

/**
 * Decide whether display text should be shown based on session status
 * Return true if text should hide
 */
export const getStatusTextHideState = createSelector(
  [getSessionStatus, getConfig],
  (status: ConnectionStatus, config: ConfigType) => {
    const autosave = config.autosave

    switch (status) {
      case ConnectionStatus.SAVING:
      case ConnectionStatus.NOTIFY_SAVED: {
        // Hide save banner if autosave is enabled
        return autosave
      }
      case ConnectionStatus.SAVED:
      case ConnectionStatus.UNSAVED:
      case ConnectionStatus.COMPUTE_DONE: {
        /**
         * Setting hide to true achieves a fade animation
         * since status transitions from NOTIFY_X to X
         * but keeps the same text
         */
        return true
      }
      default: {
        return false
      }
    }
  }
)
