import { RedisConfig } from "../types/config"
import { ItemExport } from "../types/export"
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
  /** the event name for model inference request */
  protected modelRequest: string
  /** the event name for model inference response */
  protected modelResponse: string

  /**
   * Create new publisher and subscriber clients
   *
   * @param client
   */
  constructor(client: RedisClient) {
    this.client = client
    this.registerEvent = "registerEvent"
    this.modelRequest = "modelRequest"
    this.modelResponse = "modelResponse"
  }

  /**
   * Broadcasts registration event of new socket
   *
   * @param data
   */
  public publishRegisterEvent(data: RegisterMessageType): void {
    this.client.publish(this.registerEvent, JSON.stringify(data))
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
    this.client.subscribe(this.registerEvent)
    // Make sure it's subscribed before any messages are published
    return await new Promise((resolve) => {
      this.client.on("subscribe", () => {
        resolve()
      })
    })
  }

  /**
   * Send model request to the model server
   *
   * @param data
   */
  public publishModelRequestEvent(
    data: [ItemExport[], number[], string]
  ): void {
    this.client.publish(this.modelRequest, JSON.stringify(data))
  }

  /**
   * Listens for incoming model response event
   *
   * @param handler
   */
  public async subscribeModelResponseEvent(
    handler: (channel: string, message: string) => void
  ): Promise<void> {
    this.client.on("message", handler)
    this.client.subscribe(this.modelResponse)
    // Make sure it's subscribed before any messages are published
    return await new Promise((resolve) => {
      this.client.on("subscribe", () => {
        resolve()
      })
    })
  }
}

/**
 * Make a publisher or subscriber for redis
 * Subscribers can't take other actions, so separate clients for pub and sub
 *
 * @param config
 */
export function makeRedisPubSub(config: RedisConfig): RedisPubSub {
  const client = new RedisClient(config)
  return new RedisPubSub(client)
}
