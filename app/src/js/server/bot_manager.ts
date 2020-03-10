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
   * Listen for new sessions and recreate old bots
   */
  public async listen (): Promise<Bot[]> {
    // listen for new sessions
    await this.subscriber.subscribeRegisterEvent(this.handleRegister.bind(this))
    return this.restoreBots()
  }

  /**
   * Recreate the bots stored in redis
   */
  public async restoreBots (): Promise<Bot[]> {
    const botKeys = await this.redisClient.getSetMembers(getRedisBotSet())
    const bots = []
    for (const botKey of botKeys) {
      const botData = await this.getBot(botKey)
      bots.push(this.makeBot(botData))
    }
    return bots
  }

  /**
   * Handles registration of new web sessions
   */
  public async handleRegister (
    _channel: string, message: string): Promise<BotData> {
    const data = JSON.parse(message) as RegisterMessageType
    const botData: BotData = {
      projectName: data.projectName,
      taskIndex: data.taskIndex,
      botId: '',
      address: data.address
    }

    if (data.bot || await this.checkBotExists(botData)) {
      return botData
    }

    botData.botId = uuid4()

    this.makeBot(botData)
    await this.saveBot(botData)
    return botData
  }

  /**
   * Check if a bot for the given task has already been registered
   */
  public async checkBotExists (botData: BotData): Promise<boolean> {
    const key = getRedisBotKey(botData)
    return this.redisClient.exists(key)
  }

  /**
   * Get the data for a bot that has been registered
   */
  public async getBot (key: string): Promise<BotData> {
    return JSON.parse(await this.redisClient.get(key))
  }

  /**
   * Delete the bot, marking it as unregistered
   */
  public async deleteBot (botData: BotData) {
    const key = getRedisBotKey(botData)
    await this.redisClient.del(key)
    await this.redisClient.setRemove(getRedisBotSet(), key)
  }

  /**
   * Save the data for a bot, marking it as registered
   */
  private async saveBot (botData: BotData) {
    const key = getRedisBotKey(botData)
    const value = JSON.stringify(botData)
    await this.redisClient.set(key, value)
    await this.redisClient.setAdd(getRedisBotSet(), key)
  }

  /**
   * Create a new bot user
   */
  private makeBot (botData: BotData): Bot {
    Logger.info(sprintf('Creating bot for project %s, task %d',
      botData.projectName, botData.taskIndex))
    const bot = new Bot(botData)

    const pollId = setInterval(async () => {
      await this.monitorActivity(bot, pollId)
    }, this.pollTime)
    return bot
  }

  /**
   * Kill the bot if no activity since time of last poll
   */
  private async monitorActivity (bot: Bot, pollId: NodeJS.Timeout) {
    if (bot.getActionCount() > 0) {
      bot.resetActionCount()
    } else {
      clearInterval(pollId)
      await this.deleteBot(bot.getData())
      bot.kill()
    }
  }
}
