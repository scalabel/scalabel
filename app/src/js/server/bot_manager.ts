import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { Bot } from './bot'
import Logger from './logger'
import { getRedisBotKey, getRedisBotSet } from './path'
import { RedisClient } from './redis_client'
import { RedisPubSub } from './redis_pub_sub'
import {
  BotData, RegisterMessageType,
  ServerConfig } from './types'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class BotManager {
  /** env variables */
  protected config: ServerConfig
  /** the redis message broker */
  protected subscriber: RedisPubSub
  /** the redis client for storage */
  protected redisClient: RedisClient
  /** the time in between polls that check session activity */
  protected pollTime: number

  constructor (
    config: ServerConfig, subscriber: RedisPubSub,
    redisClient: RedisClient, pollTime?: number) {
    this.config = config
    this.subscriber = subscriber
    this.redisClient = redisClient
    if (pollTime) {
      this.pollTime = pollTime
    } else {
      this.pollTime = 1000 * 60 * 5 // 5 minutes in ms
    }
  }

  /**
   * Listen for new users and recreate old ones
   */
  public async listen (): Promise<Bot[]> {
    // listen for new users
    await this.subscriber.subscribeRegisterEvent(this.handleRegister.bind(this))
    return this.restoreUsers()
  }

  /**
   * Recreate the virtual users stored in redis
   */
  public async restoreUsers (): Promise<Bot[]> {
    const webIds = await this.redisClient.getSetMembers(getRedisBotSet())
    const bots = []
    for (const webId of webIds) {
      const botData = await this.getBot(webId)
      bots.push(this.makeBot(botData))
    }
    return bots
  }

  /**
   * Handles registration of new web sessions
   */
  public async handleRegister (
    _channel: string, message: string): Promise<Bot> {
    const data = JSON.parse(message) as RegisterMessageType
    const botData: BotData = {
      webId: data.userId,
      botId: '',
      serverAddress: data.address
    }

    // if bot already exists, just return a dummy bot
    if (data.bot || await this.checkBotExists(data.userId)) {
      return new Bot(botData)
    }

    botData.botId = uuid4()

    const bot = this.makeBot(botData)
    bot.makeSession(data.projectName, data.taskIndex)
    await this.saveBot(botData)
    return bot
  }

  /**
   * Check if a bot for the given user has already been registered
   */
  public async checkBotExists (userId: string): Promise<boolean> {
    const key = getRedisBotKey(userId)
    return this.redisClient.exists(key)
  }

  /**
   * Get the data for a bot that has been registered
   */
  public async getBot (userId: string): Promise<BotData> {
    const key = getRedisBotKey(userId)
    return JSON.parse(await this.redisClient.get(key))
  }

  /**
   * Delete the bot, marking it as unregistered
   */
  public async deleteBot (userId: string) {
    const key = getRedisBotKey(userId)
    await this.redisClient.del(key)
    await this.redisClient.setRemove(getRedisBotSet(), key)
  }

  /**
   * Save the data for a bot, marking it as registered
   */
  private async saveBot (botData: BotData) {
    const key = getRedisBotKey(botData.webId)
    const value = JSON.stringify(botData)
    await this.redisClient.set(key, value)
    await this.redisClient.setAdd(getRedisBotSet(), botData.webId)
  }

  /**
   * Create a new bot user
   */
  private makeBot (botData: BotData): Bot {
    Logger.info(sprintf('Creating bot for user %s', botData.webId))
    const bot = new Bot(botData)

    const pollId = setInterval(async () => {
      await this.checkSessionActivity(bot, pollId)
    }, this.pollTime)
    return bot
  }

  /**
   * Kill the session if no activity since time of last poll
   */
  private async checkSessionActivity (bot: Bot, pollId: NodeJS.Timeout) {
    if (bot.getActionCount() > 0) {
      bot.resetActionCount()
    } else {
      clearInterval(pollId)
      await this.deleteBot(bot.webId)
      bot.kill()
    }
  }
}
