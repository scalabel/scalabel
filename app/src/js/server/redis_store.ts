import { promisify } from 'util'
import Logger from './logger'
import * as path from './path'
import { RedisClient } from './redis_client'
import { Storage } from './storage'
import { ServerConfig, StateMetadata } from './types'

/**
 * Wraps and promisifies redis key value storage functionality
 */
export class RedisStore {
  /** the key value client */
  protected client: RedisClient
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
  constructor (env: ServerConfig, storage: Storage, client: RedisClient) {
    this.timeout = env.redisTimeout
    this.timeForWrite = env.timeForWrite
    this.numActionsForWrite = env.numActionsForWrite
    this.storage = storage
    this.client = client
    this.client.config('SET', 'notify-keyspace-events', 'Ex')

    // subscribe to reminder expirations for saving
    this.client.subscribe('__keyevent@0__:expired')
    this.client.on('message', async (_channel: string, reminderKey: string) => {
      const baseKey = path.getRedisBaseKey(reminderKey)
      const metaKey = path.getRedisMetaKey(baseKey)
      const metadata: StateMetadata = JSON.parse(await this.get(metaKey))
      const saveDir = path.getSaveDir(metadata.projectName, metadata.taskId)

      const value = await this.get(baseKey)
      await this.writeBackTask(saveDir, value)
      this.del(baseKey)
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
  public async setExWithReminder (
    saveDir: string, value: string, metadata: string) {
    // Update value and metadata atomically
    const keys = [saveDir, path.getRedisMetaKey(saveDir)]
    const vals = [value, metadata]
    await this.setAtomic(keys, vals, this.timeout)
    // Handle write back to storage
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
      this.del(reminderKey)
    } else {
      // otherwise just update the action counter
      const newActions = (numActions + 1).toString()
      await this.setEx(reminderKey, newActions, this.timeForWrite)
    }
  }

  /**
   * Update multiple value atomically
   */
  public async setAtomic (keys: string[], vals: string[], timeout: number) {
    const timeoutMs = timeout * 1000
    const multi = this.client.multi()
    if (keys.length !== vals.length) {
      Logger.error(Error('Keys do not match values'))
      return
    }
    for (let i = 0; i < keys.length; i++) {
      multi.psetex(keys[i], timeoutMs, vals[i])
    }
    const multiExecAsync = promisify(multi.exec).bind(multi)
    await multiExecAsync()
  }

  /**
   * Wrapper for redis delete
   */
  public del (key: string) {
    this.client.del(key)
  }

   /**
    * Wrapper for redis get
    */
  public async get (key: string): Promise<string> {
    return this.client.get(key)
  }

   /**
    * Wrapper for redis incr
    */
  public async incr (key: string) {
    await this.client.incr(key)
  }

  /**
   * Wrapper for redis set
   */
  public async set (key: string, value: string) {
    await this.client.set(key, value)
  }

  /**
   * Wrapper for redis set with expiration
   * Private because calling directly will cause problems with missing metadata
   */
  private async setEx (key: string, value: string, timeout: number) {
    const timeoutMs = timeout * 1000
    await this.client.psetex(key, timeoutMs, value)
  }
}
