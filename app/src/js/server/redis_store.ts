import { promisify } from 'util'
import Logger from './logger'
import * as path from './path'
import { RedisClient } from './redis_client'
import { Storage } from './storage'
import { ServerConfig, StateMetadata } from './types'

/**
 * Wraps high level redis functionality
 * Including caching, atomic writes,
 * and writing back after a set time or a set number of writes
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
  constructor (config: ServerConfig, storage: Storage, client: RedisClient) {
    this.timeout = config.redisTimeout
    this.timeForWrite = config.timeForWrite
    this.numActionsForWrite = config.numActionsForWrite
    this.storage = storage
    this.client = client
    this.client.config('SET', 'notify-keyspace-events', 'Ex')

    // subscribe to reminder expirations for saving
    this.client.subscribe('__keyevent@0__:expired')
    this.client.on('message', async (_channel: string, reminderKey: string) => {
      /**
       * Check that the key is from a reminder expiring
       * Not from a normal key or meta key expiring
       */
      if (!path.checkRedisReminderKey(reminderKey)) {
        return
      }
      const baseKey = path.getRedisBaseKey(reminderKey)
      const metaKey = path.getRedisMetaKey(baseKey)
      const metadata: StateMetadata = JSON.parse(await this.get(metaKey))
      const saveDir = path.getSaveDir(metadata.projectName, metadata.taskId)

      const value = await this.get(baseKey)
      await this.writeBackTask(saveDir, value)
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
    saveDir: string, value: string, metadata: string, numActionsSaved: number) {
    // Update value and metadata atomically
    const keys = [saveDir, path.getRedisMetaKey(saveDir)]
    const vals = [value, metadata]
    await this.setAtomic(keys, vals, this.timeout)

    await this.setWriteReminder(saveDir, value, numActionsSaved)
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
   * Wrapper for get
   */
  public async get (key: string) {
    return this.client.get(key)
  }

  /**
   * Wrapper for del
   */
  public async del (key: string) {
    await this.client.del(key)
  }

  /**
   * Handle writing back to storage
   */
  private async setWriteReminder (
    saveDir: string, value: string, numActionsSaved: number) {
    const reminderKey = path.getRedisReminderKey(saveDir)
    let numActions = parseInt(await this.get(reminderKey), 10)
    if (!numActions) {
      numActions = 0
    }
    numActions += numActionsSaved

    if (numActions >= this.numActionsForWrite) {
      // write condition: num actions exceeded limit
      await this.writeBackTask(saveDir, value)
      await this.del(reminderKey)
    } else {
      // otherwise just update the action counter
      await this.setEx(reminderKey, numActions.toString(), this.timeForWrite)
    }
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
