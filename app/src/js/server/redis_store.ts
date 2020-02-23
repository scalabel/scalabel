import * as redis from 'redis'
import { promisify } from 'util'
import Logger from './logger'
import * as path from './path'
import { Storage } from './storage'
import { ServerConfig } from './types'

/**
 * Wraps and promisifies redis functionality
 */
export class RedisStore {
  /** the redis client */
  protected client: redis.RedisClient
  /** the redis client for subscribing */
  protected sub: redis.RedisClient
  /** the timeout in seconds for flushing a value */
  protected timeout: number
  /** after last update, waits this many seconds before writing to storage */
  protected timeForWrite: number
  /** writes back to storage after this number of actions for a task */
  protected numActionsForWrite: number
  /** storage to write back to */
  protected storage: Storage

  /**
   * Create new store
   */
  constructor (env: ServerConfig, storage: Storage) {
    this.timeout = env.redisTimeout
    this.timeForWrite = env.timeForWrite
    this.numActionsForWrite = env.numActionsForWrite
    this.storage = storage
    this.client = redis.createClient(env.redisPort)
    this.client.on('error', (err: Error) => {
      Logger.error(err)
    })
    this.client.on('ready', () => {
      this.client.config('SET', 'notify-keyspace-events', 'Ex')
    })

    this.sub = redis.createClient(env.redisPort)
    this.sub.on('error', (err: Error) => {
      Logger.error(err)
    })
    // subscribe to reminder expirations for saving
    this.sub.subscribe('__keyevent@0__:expired')
    this.sub.on('message', async (_channel: string, reminderKey: string) => {
      const baseKey = path.getRedisBaseKey(reminderKey)
      const saveDir = await this.get(path.getRedisMetaKey(baseKey))
      const value = await this.get(baseKey)
      await this.writeBackTask(saveDir, value)
      await this.del(baseKey)
    })
  }

  /**
   * Writes back task submission to storage
   * Task key in redis is the directory, so add a date before writing
   */
  public async writeBackTask (saveDir: string, value: string) {
    const fileKey = path.getFileKey(saveDir)
    await this.storage.save(fileKey, value)
  }

  /**
   * Sets a value with timeout for flushing
   * And a reminder value with shorter timeout for saving to disk
   */
  public async setExWithReminder (saveDir: string, value: string) {
    await this.setEx(saveDir, value, this.timeout)
    await this.setEx(path.getRedisMetaKey(saveDir), saveDir, this.timeout)

    if (this.numActionsForWrite === 1) {
      // special case: always save, don't need reminder
      await this.writeBackTask(saveDir, value)
      return
    }

    const reminderKey = path.getRedisReminderKey(saveDir)
    const numActions = parseInt(await this.get(reminderKey), 10)
    if (!numActions) {
      // new reminder, 1st action
      await this.setEx(reminderKey, '1', this.timeForWrite)
    } else if (numActions + 1 >= this.numActionsForWrite) {
      // write condition: num actions exceeded limit
      await this.writeBackTask(saveDir, value)
      await this.del(reminderKey)
    } else {
      // otherwise just update the action counter
      const newActions = (numActions + 1).toString()
      await this.setEx(reminderKey, newActions, this.timeForWrite)
    }
  }

  /**
   * Wrapper for redis delete
   */
  public async del (key: string) {
    this.client.del(key)
  }

  /**
   * Wrapper for redis set with expiration
   */
  public async setEx (key: string, value: string, timeout: number) {
    const timeoutMs = timeout * 1000
    const redisSetAsync = promisify(this.client.psetex).bind(this.client)
    await redisSetAsync(key, timeoutMs, value)
  }

   /**
    * Wrapper for redis get
    */
  public async get (key: string): Promise<string> {
    const redisGetAsync = promisify(this.client.get).bind(this.client)
    const redisValue: string = await redisGetAsync(key)
    return redisValue
  }

   /**
    * Wrapper for redis incr
    */
  public async incr (key: string) {
    const redisIncrAsync = promisify(this.client.incr).bind(this.client)
    await redisIncrAsync(key)
  }
}
