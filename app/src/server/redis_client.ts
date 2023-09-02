import { createClient, RedisClientType } from "@redis/client"

import { RedisConfig } from "../types/config"
import Logger from "./logger"

const redisClient = createClient()
type RedisClientMultiCommandType = ReturnType<typeof redisClient.multi>

/**
 * Exposes promisified versions of the necessary methods on a redis client
 * This should implement KeyValue and PubSub interfaces
 */
export class RedisClient {
  /** The redis client for standard key value ops */
  protected client: RedisClientType
  /** The redis client for pub/sub of events */
  protected pubSub: RedisClientType

  /**
   * Constructor
   *
   * @param config
   * @param withLogging
   */
  constructor(config: RedisConfig, withLogging = false) {
    this.client = createClient({
      socket: {
        port: config.port
      }
    })
    this.pubSub = createClient({
      socket: {
        port: config.port
      }
    })

    this.client.on("error", (err: Error) => {
      if (withLogging) {
        Logger.error(err)
      }
    })
    this.pubSub.on("error", (err: Error) => {
      if (withLogging) {
        Logger.error(err)
      }
    })
  }

  /** Connect redis clients to the server */
  public async setup(): Promise<void> {
    await this.client.connect().catch(async (err) => {
      return await Promise.reject(err)
    })
    await this.pubSub.connect().catch(async (err) => {
      return await Promise.reject(err)
    })
    return await Promise.resolve()
  }

  /**
   * Add a handler function
   * Note that the handler and subscriber must use the same client
   *
   * @param event
   * @param callback
   */
  public on(
    event: string,
    callback: (channel: string, value: string) => void
  ): void {
    this.pubSub.on(event, callback)
  }

  /**
   * Subscribe to a channel
   *
   * @param channel
   */
  public async subscribe(channel: string): Promise<void> {
    await this.pubSub.subscribe(channel, (message, channelName) => {
      console.log(message, channelName)
    })
  }

  /**
   * Publish to a channel
   *
   * @param channel
   * @param message
   */
  public async publish(channel: string, message: string): Promise<void> {
    await this.pubSub.publish(channel, message)
  }

  /**
   * Wrapper for redis delete
   *
   * @param key
   */
  public async del(key: string): Promise<void> {
    await this.client.del(key)
  }

  /** Start an atomic transaction */
  public multi(): RedisClientMultiCommandType {
    return this.client.multi()
  }

  /**
   * Wrapper for redis get
   *
   * @param key
   */
  public async get(key: string): Promise<string | null> {
    return await this.client.get(key)
  }

  /**
   * Wrapper for redis exists
   *
   * @param key
   */
  public async exists(key: string): Promise<boolean> {
    const exists = await this.client.exists(key)
    return await new Promise((resolve) => {
      resolve(exists !== 0)
    })
  }

  /**
   * Wrapper for redis set add
   *
   * @param key
   * @param value
   */
  public async setAdd(key: string, value: string): Promise<void> {
    await this.client.sAdd(key, value)
  }

  /**
   * Wrapper for redis set remove
   *
   * @param key
   * @param value
   */
  public async setRemove(key: string, value: string): Promise<void> {
    await this.client.sRem(key, value)
  }

  /**
   * Wrapper for redis set members
   *
   * @param key
   */
  public async getSetMembers(key: string): Promise<string[]> {
    return await this.client.sMembers(key)
  }

  /**
   * Wrapper for redis psetex
   *
   * @param key
   * @param timeout
   * @param value
   */
  public async pSetEx(
    key: string,
    timeout: number,
    value: string
  ): Promise<void> {
    await this.client.pSetEx(key, timeout, value)
  }

  /**
   * Wrapper for redis set
   *
   * @param key
   * @param value
   */
  public async set(key: string, value: string): Promise<void> {
    await this.client.set(key, value)
  }

  /**
   * This function is used to get keys with a given prefix.
   * This is useful when deleting project.
   *
   * @param prefix
   */
  public async getKeysWithPrefix(prefix: string): Promise<string[]> {
    return await this.client.keys(prefix + "*")
  }

  /**
   * Wrapper for redis config
   *
   * @param name
   * @param value
   */
  public configSet(name: string, value: string): void {
    const asyncFunc: () => void = async () => {
      await this.client.configSet(name, value)
    }
    this.client.on("ready", asyncFunc)
  }

  /** Close the connection to the server */
  public async close(): Promise<void> {
    await this.client.quit()
    await this.pubSub.quit()
  }
}
