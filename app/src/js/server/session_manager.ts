import { ServerConfig } from './types'
import { VirtualSession } from './virtual_session'

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

    // use redis pubsub to listen to
    // make sure its a saveDir, then extract the taskId
    this.makeVirtualSession()
    return
  }

  /**
   * Create and start a new virtual session
   */
  private makeVirtualSession () {
    const sess = new VirtualSession()
    sess.listen()

    // should also kill the sess if number of sockets in room is 1 (0 + self)
  }
}
