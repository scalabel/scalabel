import { ServerConfig } from './types'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class SessionManager {
  /** env variables */
  protected config: ServerConfig

  constructor (config: ServerConfig) {
    this.config = config
  }

  /**
   * Listens to redis changes
   */
  public listen () {
    return
  }
}