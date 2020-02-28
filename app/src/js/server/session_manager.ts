import { ServerConfig } from './types'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class SessionManager {
  /** env variables */
  protected config: ServerConfig

  constructor (config: ServerConfig) {
    this.config = config
    // store a map from task id to virtual session
  }

  /**
   * Listens to redis changes
   */
  public listen () {
    // when a new task is added to redis, create a virtual session
    // virtual session should use socketio channels

    // use redis keyevents to subscribe to set events
    // make sure its a saveDir, then extract the taskId
    return
  }
}