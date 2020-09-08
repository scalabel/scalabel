import * as redis from "redis"
import { promisify } from "util"

import { RedisConfig } from "../types/config"
import Logger from "./logger"

/**
 * Exposes promisified versions of the necessary methods on a redis client
 * This should implement KeyValue and PubSub interfaces
 */
export class RedisClient {
  /** The redis client for standard key value ops */
  protected client: redis.RedisClient
  /** The redis client for pub/sub of events */
  protected pubSub: redis.RedisClient

  /**
   * Constructor
   *
   * @param config
   * @param withLogging
   */
  constructor(config: RedisConfig, withLogging = false) {
    this.client = redis.createClient(config.port)
    this.pubSub = redis.createClient(config.port)

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
  public subscribe(channel: string): void {
    this.pubSub.subscribe(channel)
  }

  /**
   * Publish to a channel
   *
   * @param channel
   * @param message
   */
  public publish(channel: string, message: string): void {
    this.pubSub.publish(channel, message)
  }

  /**
   * Wrapper for redis delete
   *
   * @param key
   */
  public async del(key: string): Promise<void> {
    this.client.del(key)
  }

  /** Start an atomic transaction */
  public multi(): redis.Multi {
    return this.client.multi()
  }

  /**
   * Wrapper for redis get
   *
   * @param key
   */
  public async get(key: string): Promise<string | null> {
    const redisGetAsync = promisify(this.client.get).bind(this.client)
    const redisValue: string | null = await redisGetAsync(key)
    return redisValue
  }

  /**
   * Wrapper for redis exists
   *
   * @param key
   */
  public async exists(key: string): Promise<boolean> {
    return await new Promise((resolve) => {
      this.client.exists(key, (_err: Error | null, exists: number) => {
        if (exists === 0) {
          resolve(false)
        } else {
          resolve(true)
        }
      })
    })
  }

  /**
   * Wrapper for redis set add
   *
   * @param key
   * @param value
   */
  public async setAdd(key: string, value: string): Promise<void> {
    this.client.sadd(key, value)
  }

  /**
   * Wrapper for redis set remove
   *
   * @param key
   * @param value
   */
  public async setRemove(key: string, value: string): Promise<void> {
    this.client.srem(key, value)
  }

  /**
   * Wrapper for redis set members
   *
   * @param key
   */
  public async getSetMembers(key: string): Promise<string[]> {
    const redisSetMembersAsync = promisify(this.client.smembers).bind(
      this.client
    )
    return await redisSetMembersAsync(key)
  }

  /**
   * Wrapper for redis psetex
   *
   * @param key
   * @param timeout
   * @param value
   */
  public async psetex(
    key: string,
    timeout: number,
    value: string
  ): Promise<void> {
    const redisSetExAsync = promisify(this.client.psetex).bind(this.client)
    await redisSetExAsync(key, timeout, value)
  }

  /**
   * Wrapper for redis set
   *
   * @param key
   * @param value
   */
  public async set(key: string, value: string): Promise<void> {
    const redisSetAsync = promisify(this.client.set).bind(this.client)
    await redisSetAsync(key, value)
  }

  /**
   * Wrapper for redis config
   *
   * @param type
   * @param name
   * @param value
   */
  public config(type: string, name: string, value: string): void {
    this.client.on("ready", () => {
      this.client.config(type, name, value)
    })
  }

  /** Close the connection to the server */
  public async close(): Promise<void> {
    await promisify(this.client.quit).bind(this.client)()
    await promisify(this.pubSub.quit).bind(this.pubSub)()
  }
}
