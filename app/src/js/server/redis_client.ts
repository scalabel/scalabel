import * as redis from 'redis'
import { promisify } from 'util'
import Logger from './logger'
import { ServerConfig } from './types'

/**
 * Exposes promisified versions of the necessary methods on a redis client
 * This should implement KeyValue and PubSub interfaces
 */
export class RedisClient {
  /** The redis client for standard key value ops */
  protected client: redis.RedisClient
  /** The redis client for pub/sub of events */
  protected pubSub: redis.RedisClient

  constructor (config: ServerConfig, withLogging = false) {
    this.client = redis.createClient(config.redisPort)
    this.pubSub = redis.createClient(config.redisPort)

    this.client.on('error', (err: Error) => {
      if (withLogging) {
        Logger.error(err)
      }
    })
    this.pubSub.on('error', (err: Error) => {
      if (withLogging) {
        Logger.error(err)
      }
    })
  }

  /**
   * Add a handler function
   * Note that the handler and subscriber must use the same client
   */
  public on (event: string,
             callback: (channel: string, value: string) => void) {
    this.pubSub.on(event, callback)
  }

  /** Subscribe to a channel */
  public subscribe (channel: string) {
    this.pubSub.subscribe(channel)
  }

  /** Publish to a channel */
  public publish (channel: string, message: string) {
    this.pubSub.publish(channel, message)
  }

  /** Wrapper for redis delete */
  public async del (key: string) {
    this.client.del(key)
  }

    /** Start an atomic transaction */
  public multi (): redis.Multi {
    return this.client.multi()
  }

  /** Wrapper for redis get */
  public async get (key: string): Promise<string> {
    const redisGetAsync = promisify(this.client.get).bind(this.client)
    const redisValue: string | null = await redisGetAsync(key)
    if (redisValue === null) {
      throw new Error(`Failed to get ${key} from redis`)
    } else {
      return redisValue
    }
  }

  /** Wrapper for redis exists */
  public async exists (key: string): Promise<boolean> {
    return new Promise((resolve, _reject) => {
      this.client.exists(key, (_err: Error | null, exists: number) => {
        if (exists === 0) {
          resolve(false)
        } else {
          resolve(true)
        }
      })
    })
  }

  /** Wrapper for redis set add */
  public async setAdd (key: string, value: string) {
    this.client.sadd(key, value)
  }

  /** Wrapper for redis set remove */
  public async setRemove (key: string, value: string) {
    this.client.srem(key, value)
  }

  /** Wrapper for redis set members */
  public async getSetMembers (key: string): Promise<string[]> {
    const redisSetMembersAsync =
      promisify(this.client.smembers).bind(this.client)
    return redisSetMembersAsync(key)
  }

  /** Wrapper for redis psetex */
  public async psetex (key: string, timeout: number, value: string) {
    const redisSetExAsync = promisify(this.client.psetex).bind(this.client)
    await redisSetExAsync(key, timeout, value)
  }

  /** Wrapper for redis set */
  public async set (key: string, value: string) {
    const redisSetAsync = promisify(this.client.set).bind(this.client)
    await redisSetAsync(key, value)
  }

  /** Wrapper for redis config */
  public config (type: string, name: string, value: string) {
    this.client.on('ready', () => {
      this.client.config(type, name, value)
    })
  }

  /** Close the connection to the server */
  public async close () {
    await promisify(this.client.quit).bind(this.client)()
    await promisify(this.pubSub.quit).bind(this.pubSub)()
  }
}
