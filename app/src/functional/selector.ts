import { createSelector } from "reselect"

import { ReduxState } from "../types/redux"
import { ConfigType, ConnectionStatus, SessionType } from "../types/state"

/**
 * Load the task config
 *
 * @param state
 */
export function getConfig(state: ReduxState): ConfigType {
  return state.present.task.config
}

/**
 * Get whether autosave is enabled
 */
export const getAutosaveFlag = createSelector(
  [getConfig],
  (config: ConfigType) => {
    return config.autosave
  }
)

/**
 * Get title of main page
 */
export const getPageTitle = createSelector(
  [getConfig],
  (config: ConfigType) => {
    return config.pageTitle
  }
)

/**
 * Get link to instructions
 */
export const getInstructionLink = createSelector(
  [getConfig],
  (config: ConfigType) => {
    return config.instructionPage
  }
)

/**
 * Get link to the dashboard
 */
export const getDashboardLink = createSelector(
  [getConfig],
  (config: ConfigType) => {
    return `/vendor?project_name=${config.projectName}`
  }
)

/**
 * Load the session
 *
 * @param state
 */
export function getSession(state: ReduxState): SessionType {
  return state.present.session
}

/**
 * Get the status of the session
 */
export const getSessionStatus = createSelector(
  [getSession],
  (session: SessionType) => {
    return session.status
  }
)

/**
 * Get the number of times the session status has updated
 */
export const getNumStatusUpdates = createSelector(
  [getSession],
  (session: SessionType) => {
    return session.numUpdates
  }
)

/**
 * Get display text based on the session status
 */
export const getStatusText = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    switch (status) {
      case ConnectionStatus.SAVING: {
        return "Saving in progress..."
      }
      case ConnectionStatus.RECONNECTING: {
        return "Trying to reconnect..."
      }
      case ConnectionStatus.COMPUTING: {
        return "Model predictions in progress.."
      }
      case ConnectionStatus.SUBMITTING: {
        return "Submitting..."
      }
      case ConnectionStatus.COMPUTE_DONE:
      case ConnectionStatus.NOTIFY_COMPUTE_DONE: {
        return "Model predictions complete."
      }
      case ConnectionStatus.SUBMITTED:
      case ConnectionStatus.NOTIFY_SUBMITTED: {
        return "Submission complete."
      }
      case ConnectionStatus.SAVED:
      case ConnectionStatus.NOTIFY_SAVED: {
        return "All progress saved."
      }
      default: {
        return "All progress saved."
      }
    }
  }
)

/**
 * Decide whether display text should be shown based on session status
 * Return true if text should hide
 */
export const shouldStatusTextHide = createSelector(
  [getSessionStatus, getAutosaveFlag],
  (status: ConnectionStatus, autosave: boolean) => {
    switch (status) {
      case ConnectionStatus.SAVING:
      case ConnectionStatus.NOTIFY_SAVED: {
        // Hide save banner if autosave is enabled
        return autosave
      }
      case ConnectionStatus.SAVED:
      case ConnectionStatus.UNSAVED:
      case ConnectionStatus.COMPUTE_DONE:
      case ConnectionStatus.SUBMITTED: {
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

/**
 * Returns true if canvas should be frozne
 */
export const shouldCanvasFreeze = createSelector(
  [getSessionStatus, getAutosaveFlag],
  (status: ConnectionStatus, autosave: boolean) => {
    return status === ConnectionStatus.RECONNECTING && autosave
  }
)

/**
 * Returns true if there is unsaved work
 */
export const isStatusUnsaved = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return status === ConnectionStatus.UNSAVED
  }
)

/**
 * Returns true if saving in progress
 */
export const isStatusSaving = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return status === ConnectionStatus.SAVING
  }
)

/**
 * Returns true if all work is saved
 */
export const isStatusSaved = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return (
      status === ConnectionStatus.SAVED ||
      status === ConnectionStatus.NOTIFY_SAVED
    )
  }
)

/**
 * Returns true if computing in progress
 */
export const isStatusComputing = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return status === ConnectionStatus.COMPUTING
  }
)

/**
 * Returns true if compute is done
 */
export const isComputeDone = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return (
      status === ConnectionStatus.COMPUTE_DONE ||
      status === ConnectionStatus.NOTIFY_COMPUTE_DONE
    )
  }
)

/**
 * Returns true if reconnection is in progress
 */
export const isStatusReconnecting = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    return status === ConnectionStatus.RECONNECTING
  }
)

/**
 * Returns false if there could be unsaved work in progress
 */
export const isSessionFullySaved = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    switch (status) {
      case ConnectionStatus.RECONNECTING:
      case ConnectionStatus.SAVING:
      case ConnectionStatus.UNSAVED: {
        return false
      }
      default: {
        return true
      }
    }
  }
)

/**
 * Returns false if some session status event is ongoing
 */
export const isSessionStatusStable = createSelector(
  [getSessionStatus],
  (status: ConnectionStatus) => {
    switch (status) {
      case ConnectionStatus.RECONNECTING:
      case ConnectionStatus.SAVING:
      case ConnectionStatus.COMPUTING: {
        return false
      }
      default: {
        return true
      }
    }
  }
)
