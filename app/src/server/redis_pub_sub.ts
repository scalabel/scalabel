import { RegisterMessageType } from "../types/message"
import { RedisClient } from "./redis_client"

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
   *
   * @param client
   */
  constructor(client: RedisClient) {
    this.client = client
    this.registerEvent = "registerEvent"
  }

  /**
   * Broadcasts registration event of new socket
   *
   * @param data
   */
  public async publishRegisterEvent(data: RegisterMessageType): Promise<void> {
    await this.client.publish(this.registerEvent, JSON.stringify(data))
  }

  /**
   * Listens for incoming registration events
   *
   * @param handler
   */
  public async subscribeRegisterEvent(
    handler: (channel: string, message: string) => void
  ): Promise<void> {
    this.client.on("message", handler)
    await this.client.subscribe(this.registerEvent)
    // Make sure it's subscribed before any messages are published
    return await new Promise((resolve) => {
      this.client.on("subscribe", () => {
        resolve()
      })
    })
  }
}
