import { RedisClient } from './redis_client'
import { RegisterMessageType } from './types'

/**
 * Wraps redis pub/sub functionality
 */
export class RedisPubSub {
  /** the pubsub client */
  protected client: RedisClient
  /** the event name for socket registration */
  protected registerEvent: string

  /**
   * Create new publisher and subscriber clients
   */
  constructor (client: RedisClient) {
    this.client = client
    this.registerEvent = 'registerEvent'
  }

  /**
   * Broadcasts registration event of new socket
   */
  public publishRegisterEvent (data: RegisterMessageType) {
    this.client.publish(this.registerEvent, JSON.stringify(data))
  }

  /**
   * Listens for incoming registration events
   */
  public subscribeRegisterEvent (
    handler: (channel: string, message: string) => void) {
    this.client.on('message', handler)
    this.client.subscribe(this.registerEvent)
  }
}
