import * as redis from 'redis'
import { promisify } from 'util'
import Logger from './logger'
import { getRedisBaseKey, pathRedisKey, reminderRedisKey } from './path'
import Session from './server_session'

/**
 * Wraps and promisifies redis functionality for caching
 */
export class RedisCache {
  /** the redis client for caching */
  protected client: redis.RedisClient
  /** the redis client for subscribing */
  protected sub: redis.RedisClient
  /** the timeout for flushing a value */
  protected timeout: number
  /** after last update, waits this long before writing back to storage */
  protected timeForWrite: number
  /** writes back to storage after this number of actions for a task */
  protected numActionsForWrite: number

  /**
   * Create new cache
   */
  constructor () {
    const env = Session.getEnv()
    this.client = redis.createClient()
    this.client.on('error', (err) => {
      Logger.error(err)
    })
    this.client.on('ready', () => {
      this.client.config('SET', 'notify-keyspace-events', 'Ex')
    })

    this.sub = redis.createClient()
    // subscribe to reminder expirations for saving
    this.sub.subscribe('__keyevent@0__:expired')
    this.sub.on('message', async (_channel: string, message: string) => {
      const key = getRedisBaseKey(message)
      const filePath = await this.get(pathRedisKey(key))
      const value = await this.get(key)
      await Session.getStorage().save(filePath, value)
    })

    this.timeout = env.redisTimeout
    this.timeForWrite = env.timeForWrite
    this.numActionsForWrite = env.numActionsForWrite
  }

  /**
   * Sets a value with timeout for flushing
   * And a reminder value with shorter timeout for saving to disk
   */
  public async setExWithReminder (
    key: string, filePath: string, value: string) {
    await this.setEx(key, value, this.timeout)
    await this.setEx(pathRedisKey(key), filePath, this.timeout)

    if (this.numActionsForWrite === 1) {
      // special case: always save, don't need reminder
      await Session.getStorage().save(filePath, value)
    } else {
      const reminderKey = reminderRedisKey(key)
      const numActions = parseInt(await this.get(reminderKey), 10)
      if (!numActions) {
        // new reminder, 1st action
        await this.setEx(reminderKey, '1', this.timeForWrite)
      } else if (numActions + 1 >= this.numActionsForWrite) {
        // write condition: num actions exceeded limit
        await Session.getStorage().save(filePath, value)
        await this.del(reminderKey)
      } else {
        // otherwise just update the action counter
        const newActions = (numActions + 1).toString()
        await this.setEx(reminderKey, newActions, this.timeForWrite)
      }
    }
  }

  /**
   * Wrapper for redis delete
   */
  public async del (key: string) {
    // const redisDelete = promisify(this.client.del).bind(this.client)
    // await redisDelete(key)
    this.client.del(key)
  }

  /**
   * Wrapper for redis set with expiration
   */
  public async setEx (key: string, value: string, timeout: number) {
    const redisSetAsync = promisify(this.client.setex).bind(this.client)
    await redisSetAsync(key, timeout, value)
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