const enum ConnectionStatus {
  NOTIFY_SAVED, SAVED, SAVING, RECONNECTING, UNSAVED,
  COMPUTING, COMPUTE_DONE, NOTIFY_COMPUTE_DONE
}

/**
 * Tracks and handles updates to the session status
 * Also forces updates of dependent view components
 */
export class SessionStatus {
  /** Current connection status */
  public status: ConnectionStatus
  /** Previous connection status */
  public prevStatus: ConnectionStatus
  /** Number of times status has changed */
  public numberOfUpdates: number
  /** Callbacks for updating the display when status changes */
  private displayCallbacks: Array<() => void>

  constructor () {
    this.status = ConnectionStatus.UNSAVED
    this.prevStatus = ConnectionStatus.UNSAVED
    this.numberOfUpdates = 0
    this.displayCallbacks = []
  }

  /**
   * Update the status, then update display
   * @param {ConnectionStatus} newStatus: new value of status
   */
  public update (newStatus: ConnectionStatus): ConnectionStatus {
    this.prevStatus = this.status
    this.status = newStatus
    // update mod 1000 since only nearby differences are important, not total
    this.numberOfUpdates = (this.numberOfUpdates + 1) % 1000
    for (const callback of this.displayCallbacks) {
      callback()
    }
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
   * Mark status as saving
   */
  public setAsSaving () {
    // Computing status overrides saving status
    if (this.status !== ConnectionStatus.COMPUTING) {
      this.update(ConnectionStatus.SAVING)
    }
  }

  /**
   * Mark status as reconnecting
   */
  public setAsReconnecting () {
    this.update(ConnectionStatus.RECONNECTING)
  }

  /**
   * Mark status as unsaved
   */
  public setAsUnsaved () {
    // If some other event is in progress, don't change it
    if (this.status !== ConnectionStatus.RECONNECTING
      && this.status !== ConnectionStatus.SAVING
      && this.status !== ConnectionStatus.COMPUTING) {
      this.update(ConnectionStatus.UNSAVED)
    }
  }

  /**
   * After a connect/reconnect, mark status as unsaved
   * regardless of previous status
   */
  public setAsConnect () {
    this.update(ConnectionStatus.UNSAVED)
  }

  /**
   * Mark status as computing
   */
  public setAsComputing () {
    this.update(ConnectionStatus.COMPUTING)
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

  /**
   * Check if unsaved
   */
  public checkUnsaved () {
    return this.status === ConnectionStatus.UNSAVED
  }

  /**
   * Check if saving
   */
  public checkSaving () {
    return this.status === ConnectionStatus.SAVING
  }

  /**
   * Check if saved
   */
  public checkSaved () {
    return this.status === ConnectionStatus.SAVED ||
      this.status === ConnectionStatus.NOTIFY_SAVED
  }

  /**
   * Check if reconnecting
   */
  public checkReconnecting () {
    return this.status === ConnectionStatus.RECONNECTING
  }

  /**
   * Check if computing
   */
  public checkComputing () {
    return this.status === ConnectionStatus.COMPUTING
  }

  /**
   * Check if model computation is done
   */
  public checkComputeDone () {
    return this.status === ConnectionStatus.COMPUTE_DONE ||
      this.status === ConnectionStatus.NOTIFY_COMPUTE_DONE
  }

  /**
   * Add a callback function for updating a display component
   */
  public addDisplayCallback (callback: () => void) {
    this.displayCallbacks.push(callback)
  }

  /**
   * Remove all display callbacks
   */
  public clearDisplayCallbacks () {
    this.displayCallbacks = []
  }

  /** Select display text based on connection status */
  public getStatusText (): string {
    switch (this.status) {
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

  /** Decide whether display text should be shown */
  public shouldStatusHide (autosave: boolean): boolean {
    switch (this.status) {
      case ConnectionStatus.SAVING:
      case ConnectionStatus.NOTIFY_SAVED: {
        if (autosave) {
          return true
        }
        return false
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

  /** Return false if there could be unsaved work in progress */
  public isFullySaved (): boolean {
    switch (this.status) {
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

  /** Return true if canvas should be frozen */
  public shouldFreezeCanvas (): boolean {
    return this.status === ConnectionStatus.RECONNECTING
  }
}
