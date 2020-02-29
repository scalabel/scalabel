import { KeyValueClient } from './interfaces'

/**
 * Wraps redis pub/sub functionality
 */
export class RedisPubSub {
  /** the pubsub client */
  protected client: KeyValueClient

  /**
   * Create new publisher and subscriber clients
   */
  constructor (client: KeyValueClient) {
    this.client = client
  }

  /**
   * Broadcasts registration event of new socket
   */
  public publishRegistrationEvent (_projectName: string, _taskId: string) {
    return
  }
}
