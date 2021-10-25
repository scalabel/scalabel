import Logger from "./logger"
import { RedisConfig } from "../types/config"
import {
  ModelKillMessageType,
  ModelRegisterMessageType,
  ModelRequestMessageType,
  RegisterMessageType
} from "../types/message"
import { RedisClient } from "./redis_client"
import { RedisChannel } from "../const/connection"

/**
 * Wraps redis pub/sub functionality
 */
export class RedisPubSub {
  /** the pubsub client */
  protected client: RedisClient

  /**
   * Create new publisher and subscriber clients
   *
   * @param client
   */
  constructor(client: RedisClient) {
    this.client = client
  }

  /**
   * Broadcasts registration event of new socket
   *
   * @param channel
   * @param data
   */
  public publishEvent(
    channel: string,
    data:
      | RegisterMessageType
      | ModelRegisterMessageType
      | ModelRequestMessageType
      | ModelKillMessageType
  ): void {
    switch (channel) {
      case RedisChannel.REGISTER_EVENT:
        data = data as RegisterMessageType
        this.client.publish(channel, JSON.stringify(data))
        break
      case RedisChannel.MODEL_REGISTER: {
        data = data as ModelRegisterMessageType
        this.client.publish(channel, JSON.stringify(data))
        break
      }
      case RedisChannel.MODEL_REQUEST: {
        data = data as ModelRequestMessageType
        const projectName: string = data.projectName
        const taskId: string = data.taskId
        channel = `${RedisChannel.MODEL_REQUEST}_${projectName}_${taskId}`
        this.client.publish(channel, JSON.stringify(data))
        break
      }
      case RedisChannel.MODEL_KILL: {
        data = data as ModelKillMessageType
        this.client.publish(channel, JSON.stringify(data))
        break
      }
      default:
        Logger.info(`Channel ${channel} is not valid!`)
    }
  }

  /**
   * Listens for incoming registration events
   *
   * @param channel
   * @param handler
   */
  public async subscribeEvent(
    channel: string,
    handler: (channel: string, message: string) => void
  ): Promise<void> {
    this.client.on("message", handler)
    this.client.subscribe(channel)
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
