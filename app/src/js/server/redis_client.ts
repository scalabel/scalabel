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

  constructor (config: ServerConfig) {
    this.client = redis.createClient(config.redisPort)
    this.client.on('error', (err: Error) => {
      Logger.error(err)
    })
    this.pubSub = redis.createClient(config.redisPort)
    this.pubSub.on('error', (err: Error) => {
      Logger.error(err)
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
    const redisValue: string = await redisGetAsync(key)
    return redisValue
  }

  /** Wrapper for redis psetex */
  public async psetex (key: string, timeout: number, value: string) {
    const redisSetExAsync = promisify(this.client.psetex).bind(this.client)
    await redisSetExAsync(key, timeout, value)
  }

  /** Wrapper for redis incr */
  public async incr (key: string) {
    const redisIncrAsync = promisify(this.client.incr).bind(this.client)
    await redisIncrAsync(key)
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
}
