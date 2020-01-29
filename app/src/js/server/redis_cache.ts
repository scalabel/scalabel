import * as redis from 'redis'
import { promisify } from 'util'
import Logger from './logger'
import { getFileKey, redisBaseKey, redisMetaKey, redisReminderKey } from './path'
import Session from './server_session'

/**
 * Wraps and promisifies redis functionality for caching
 */
export class RedisCache {
  /** the redis client for caching */
  protected client: redis.RedisClient
  /** the redis client for subscribing */
  protected sub: redis.RedisClient
  /** the timeout in seconds for flushing a value */
  protected timeout: number
  /** after last update, waits this many seconds before writing to storage */
  protected timeForWrite: number
  /** writes back to storage after this number of actions for a task */
  protected numActionsForWrite: number

  /**
   * Create new cache
   */
  constructor (port = 6379) {
    const env = Session.getEnv()
    this.client = redis.createClient(port)
    this.client.on('error', (err: Error) => {
      Logger.error(err)
    })
    this.client.on('ready', () => {
      this.client.config('SET', 'notify-keyspace-events', 'Ex')
    })

    this.sub = redis.createClient(port)
    // subscribe to reminder expirations for saving
    this.sub.subscribe('__keyevent@0__:expired')
    this.sub.on('message', async (_channel: string, message: string) => {
      const key = redisBaseKey(message)
      const metadata = await this.get(redisMetaKey(key))
      const saveKey = metadata[0]

      const value = await this.get(key)
      const fileKey = getFileKey(saveKey)
      await Session.getStorage().save(fileKey, value)
      await this.del(key)
    })

    // convert from seconds to ms
    this.timeout = env.redisTimeout * 1000
    this.timeForWrite = env.timeForWrite * 1000
    this.numActionsForWrite = env.numActionsForWrite
  }

  /**
   * Sets a value with timeout for flushing
   * And a reminder value with shorter timeout for saving to disk
   */
  public async setExWithReminder (
    saveKey: string, value: string, metadata: string) {
    const allMetadata = JSON.stringify([saveKey, metadata])

    // Update value and metadata atomically
    const keys = [saveKey, redisMetaKey(saveKey)]
    const vals = [value, allMetadata]
    await this.setAtomic(keys, vals, this.timeout)

    // Handle write back to storage
    if (this.numActionsForWrite === 1) {
      // special case: always save, don't need reminder
      const fileKey = getFileKey(saveKey)
      await Session.getStorage().save(fileKey, value)
    } else {
      const reminderKey = redisReminderKey(saveKey)
      const numActions = parseInt(await this.get(reminderKey), 10)
      if (!numActions) {
        // new reminder, 1st action
        await this.setEx(reminderKey, '1', this.timeForWrite)
      } else if (numActions + 1 >= this.numActionsForWrite) {
        // write condition: num actions exceeded limit
        const fileKey = getFileKey(saveKey)
        await Session.getStorage().save(fileKey, value)
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
    this.client.del(key)
  }

  /**
   * Update multiple value atomically
   */
  public async setAtomic (keys: string[], vals: string[], timeout: number) {
    const multi = this.client.multi()
    if (keys.length !== vals.length) {
      Logger.error(Error('Keys do not match values'))
      return
    }
    for (let i = 0; i < keys.length; i++) {
      multi.psetex(keys[i], timeout, vals[i])
    }
    const multiExecAsync = promisify(multi.exec).bind(multi)
    await multiExecAsync()
  }

  /**
   * Wrapper for redis set with expiration
   */
  public async setEx (key: string, value: string, timeout: number) {
    const redisSetAsync = promisify(this.client.psetex).bind(this.client)
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
