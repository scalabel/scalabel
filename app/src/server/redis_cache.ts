import { promisify } from "util"

import { RedisConfig } from "../types/config"
import Logger from "./logger"
import * as path from "./path"
import { RedisClient } from "./redis_client"
import { Storage } from "./storage"

/**
 * If the key is temporary and doesn't need writeback,
 *
 * @param key
 */
function isTempKey(key: string): boolean {
  return key.search(":") !== -1
}

/**
 * Wraps high level redis functionality
 * Including caching, atomic writes,
 * and writing back after a set time or a set number of writes
 */
export class RedisCache {
  /** the key value client */
  protected client: RedisClient
  /** after last update, waits this many seconds before writing to storage */
  protected writebackTime: number
  /** Cache writing limit before writing to storage */
  protected writebackCount: number
  /** storage to write back to */
  protected storage: Storage

  /**
   * Create new store
   *
   * @param config
   * @param storage
   * @param client
   */
  constructor(config: RedisConfig, storage: Storage, client: RedisClient) {
    this.writebackTime = config.writebackTime
    this.writebackCount = config.writebackCount
    this.storage = storage
    this.client = client
    this.client.config("SET", "notify-keyspace-events", "Ex")

    // Subscribe to reminder expirations for saving
    this.client.subscribe("__keyevent@0__:expired")
    this.client.on(
      "message",
      // The .on() function argument type caused the lint error
      // eslint-disable-next-line @typescript-eslint/no-misused-promises
      async (_channel: string, reminderKey: string): Promise<void> => {
        await this.processExpiredKey(reminderKey)
      }
    )
  }

  /**
   * Update multiple value atomically
   *
   * @param keys
   * @param values
   */
  public async setMulti(keys: string[], values: string[]): Promise<void> {
    if (keys.length !== values.length) {
      Logger.error(Error("Keys do not match values"))
      return
    }
    const multi = this.client.multi()
    const setArgs: Array<[string, string, number]> = []
    for (let i = 0; i < keys.length; i++) {
      setArgs.push(...(await this.makeSetArgs(keys[i], values[i])))
    }
    setArgs.map((v) => {
      if (v[2] > 0) {
        multi.psetex(v[0], v[2] * 1000, v[1])
      } else {
        multi.set(v[0], v[1])
      }
    })
    const multiExecAsync = promisify(multi.exec).bind(multi)
    await multiExecAsync()
  }

  /**
   * Wrapper for get
   *
   * @param key
   * @param checkStorage
   */
  public async get(
    key: string,
    checkStorage: boolean = true
  ): Promise<string | null> {
    let result = await this.client.get(key)
    if (checkStorage && result === null && !isTempKey(key)) {
      // If the key doesn't exit in redis, it may be evicted or the redis
      // server was relaunched. Try the storage.
      Logger.info(`Failed to get the key ${key}. Searching storage.`)
      result = await this.storage.getWithBackup(key)
      if (result !== null) {
        Logger.info(`Found ${key} in storage. Saving in redis.`)
        await this.set(key, result)
      }
    }
    return result
  }

  /**
   * Wrapper for del
   *
   * @param key
   */
  public async del(key: string): Promise<void> {
    await this.client.del(key)
  }

  /**
   * Writes back task submission to storage
   * Task key in redis is the directory, so add a date before writing
   *
   * @param key
   * @param value
   */
  public async writeback(key: string, value: string): Promise<void> {
    Logger.info(`Writing back ${key}`)
    await this.storage.saveWithBackup(key, value)
  }

  /**
   * Cache key value
   *
   * @param key
   * @param value
   */
  public async set(key: string, value: string): Promise<void> {
    Logger.debug(`Redis cache set ${key}`)
    await this.setMulti([key], [value])
  }

  /**
   * Make the arguments for redis client set from key and value
   * The return type is in [key, value, timeout]
   * timeout is in seconds. Negative timeout means no timeout
   *
   * @param key
   * @param value
   */
  private async makeSetArgs(
    key: string,
    value: string
  ): Promise<Array<[string, string, number]>> {
    const args: Array<[string, string, number]> = []
    if (
      !isTempKey(key) &&
      (this.writebackTime > 0 || this.writebackCount > 0)
    ) {
      const reminderKey = path.getRedisReminderKey(key)
      const counterValue = await this.get(reminderKey)
      let counter = 0
      if (counterValue !== null) {
        counter = parseInt(counterValue, 10)
      }
      counter += 1
      if (this.writebackCount > 0 && counter >= this.writebackCount) {
        // When the writing counter is greater than cacheLimit,
        // write back to storage
        this.writeback(key, value)
          .then(
            () => {},
            () => {}
          )
          .catch(() => {})
        counter = 0
      }
      args.push([reminderKey, counter.toString(), this.writebackTime])
    }
    args.push([key, value, -1])
    return args
  }

  /**
   * Check that the key is from a reminder expiring
   * Not from a normal key or meta key expiring
   *
   * @param reminderKey
   */
  private async processExpiredKey(reminderKey: string): Promise<void> {
    if (!path.checkRedisReminderKey(reminderKey)) {
      return
    }
    const baseKey = path.getRedisBaseKey(reminderKey)
    const value = await this.get(baseKey, false)
    if (value === null) {
      Logger.info(`Failed to get expired key ${baseKey}`)
      return
    }
    await this.writeback(baseKey, value)
  }
}
