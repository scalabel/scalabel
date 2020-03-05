import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { BotUser } from './bot_user'
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
    config: ServerConfig, subscriber: RedisPubSub, redisClient: RedisClient) {
    this.config = config
    this.subscriber = subscriber
    this.redisClient = redisClient
    this.pollTime = 1000 * 60 * 5 // 5 minutes in ms

    // listen for new users
    this.subscriber.subscribeRegisterEvent(this.handleRegister.bind(this))
  }

  /**
   * Recreate the virtual users stored in redis
   */
  public async restoreUsers () {
    const botKeys = await this.redisClient.getSetMembers(getRedisBotSet())
    for (const botKey of botKeys) {
      const botData = await this.getBot(botKey)
      this.makeBotUser(botData)
      // todo: somehow restore the sessions for the bot user
    }
  }

  /**
   * Handles registration of new web sessions
   */
  public async handleRegister (_channel: string, message: string) {
    const data = JSON.parse(message) as RegisterMessageType

    if (data.bot || await this.checkBotExists(data.userId)) {
      return
    }

    const botData: BotData = {
      webId: data.userId,
      botId: uuid4(),
      serverAddress: data.address
    }

    const bot = this.makeBotUser(botData)
    bot.makeSession(data.projectName, data.taskIndex)
    await this.saveBot(botData)
  }

  /**
   * Check if a bot for the given user has already been registered
   */
  private async checkBotExists (userId: string): Promise<boolean> {
    const key = getRedisBotKey(userId)
    return this.redisClient.exists(key)
  }

  /**
   * Get the data for a bot that has been registered
   */
  private async getBot (key: string): Promise<BotData> {
    return JSON.parse(await this.redisClient.get(key))
  }

  /**
   * Save the data for a bot, marking it as registered
   */
  private async saveBot (botData: BotData) {
    const key = getRedisBotKey(botData.webId)
    const value = JSON.stringify(botData)
    await this.redisClient.set(key, value)
    await this.redisClient.setAdd(getRedisBotSet(), key)
  }

  /**
   * Delete the bot, marking it as unregistered
   */
  private async deleteBot (userId: string) {
    const key = getRedisBotKey(userId)
    await this.redisClient.del(key)
    await this.redisClient.setRemove(getRedisBotSet(), key)
  }

  /**
   * Create a new bot user
   */
  private makeBotUser (botData: BotData): BotUser {
    Logger.info(sprintf('Creating bot for user %s', botData.webId))
    const bot = new BotUser(botData)

    const pollId = setInterval(async () => {
      await this.checkSessionActivity(bot, pollId)
    }, this.pollTime)
    return bot
  }

  /**
   * Kill the session if no activity since time of last poll
   */
  private async checkSessionActivity (bot: BotUser, pollId: NodeJS.Timeout) {
    if (bot.getActionCount() > 0) {
      bot.resetActionCount()
    } else {
      clearInterval(pollId)
      await this.deleteBot(bot.webId)
      bot.kill()
    }
  }
}
