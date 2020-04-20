import { updateSessionStatus } from '../action/common'
import { ConnectionStatus } from '../functional/types'
import Session from './session'

/**
 * Tracks and handles updates to the session status
 * Also forces updates of dependent view components
 */
export class SessionStatus {
  /** Current connection status */
  public status: ConnectionStatus
  /** Number of times status has changed */
  public numberOfUpdates: number

  constructor () {
    this.status = ConnectionStatus.UNSAVED
    this.numberOfUpdates = 0
  }

  /**
   * Update the status, then update display
   * @param {ConnectionStatus} newStatus: new value of status
   */
  public update (newStatus: ConnectionStatus): ConnectionStatus {
    Session.dispatch(updateSessionStatus(newStatus))
    return newStatus
  }

  /**
   * Update status after waiting a specified time
   * Only update if status hasn't changed since then
   */
  public waitThenUpdate (newStatus: ConnectionStatus, seconds: number) {
    const numberOfUpdates = this.numberOfUpdates
    setTimeout(() => {
      // Don't update if other effect occurred in between
      if (this.numberOfUpdates === numberOfUpdates) {
        this.update(newStatus)
      }
    }, seconds * 1000)
  }
  
  /**
   * Mark status as compute done
   */
  public setAsComputeDone () {
    // After 5 seconds, compute done message fades out
    this.update(ConnectionStatus.NOTIFY_COMPUTE_DONE)
    this.waitThenUpdate(ConnectionStatus.COMPUTE_DONE, 5)
  }

  /**
   * Mark status as saved
   */
  public setAsSaved () {
    if (this.status !== ConnectionStatus.COMPUTING) {
      // After 5 seconds, saved message fades out
      this.update(ConnectionStatus.NOTIFY_SAVED)
      this.waitThenUpdate(ConnectionStatus.SAVED, 5)
    }
  }
}
