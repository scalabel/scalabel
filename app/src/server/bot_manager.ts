import { uid } from "../common/uid"
import { RedisChannel } from "../const/connection"
import { ServerConfig } from "../types/config"
import {
  BotData,
  ModelStatusMessageType,
  RegisterMessageType
} from "../types/message"
import { Bot } from "./bot"
import Logger from "./logger"
import { getRedisBotKey, getRedisBotSet } from "./path"
import { RedisClient } from "./redis_client"
import { RedisPubSub } from "./redis_pub_sub"
import { index2str } from "../common/util"

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class BotManager {
  /** env variables */
  protected config: ServerConfig
  /** the redis message broker */
  protected publisher: RedisPubSub
  /** the redis message broker */
  protected subscriber: RedisPubSub
  /** the redis client for storage */
  protected redisClient: RedisClient
  /** the time in between polls that check session activity */
  protected pollTime: number

  /**
   * Constructor
   *
   * @param config
   * @param publisher
   * @param subscriber
   * @param redisClient
   * @param pollTime
   */
  constructor(
    config: ServerConfig,
    publisher: RedisPubSub,
    subscriber: RedisPubSub,
    redisClient: RedisClient,
    pollTime?: number
  ) {
    this.config = config
    this.publisher = publisher
    this.subscriber = subscriber
    this.redisClient = redisClient
    if (pollTime !== undefined) {
      this.pollTime = pollTime
    } else {
      this.pollTime = 5 * 1000 * 60 // 5 minutes in ms
    }
  }

  /**
   * Listen for new sessions and recreate old bots
   */
  public async listen(): Promise<BotData[]> {
    // Listen for new sessions
    await this.subscriber.subscribeEvent(
      RedisChannel.REGISTER_EVENT,
      // eslint-disable-next-line @typescript-eslint/no-misused-promises
      this.handleRegister.bind(this)
    )
    return await this.restoreBots()
  }

  /**
   * Recreate the bots stored in redis
   */
  public async restoreBots(): Promise<Bot[]> {
    const botKeys = await this.redisClient.getSetMembers(getRedisBotSet())
    const bots: Bot[] = []
    for (const botKey of botKeys) {
      const botData = await this.getBot(botKey)
      const bot = this.makeBot(botData)
      await bot.listen()
      bots.push(bot)
    }
    return bots
  }

  /**
   * Handles registration of new web sessions
   *
   * @param _channel
   * @param message
   */
  public async handleRegister(
    _channel: string,
    message: string
  ): Promise<BotData> {
    const data = JSON.parse(message) as RegisterMessageType
    const botData: BotData = {
      projectName: data.projectName,
      taskIndex: data.taskIndex,
      botId: "",
      address: data.address,
      labelType: data.labelType !== undefined ? data.labelType : "box2d"
    }

    if (data.bot || (await this.checkBotExists(botData))) {
      const modelStatusMessage: ModelStatusMessageType = {
        projectName: botData.projectName,
        taskId: index2str(botData.taskIndex),
        active: true
      }
      this.publisher.publishEvent(RedisChannel.MODEL_STATUS, modelStatusMessage)
      return botData
    }
    botData.botId = uid()

    const bot = this.makeBot(botData)
    await bot.listen()
    await this.saveBot(botData)
    return botData
  }

  /**
   * Check if a bot for the given task has already been registered
   *
   * @param botData
   */
  public async checkBotExists(botData: BotData): Promise<boolean> {
    const key = getRedisBotKey(botData)
    return await this.redisClient.exists(key)
  }

  /**
   * Get the data for a bot that has been registered
   *
   * @param key
   */
  public async getBot(key: string): Promise<BotData> {
    const data = await this.redisClient.get(key)
    if (data === null) {
      throw new Error(`Failed to get bot ${key}`)
    }
    return JSON.parse(data)
  }

  /**
   * Delete the bot, marking it as unregistered
   *
   * @param botData
   */
  public async deleteBot(botData: BotData): Promise<void> {
    const key = getRedisBotKey(botData)
    await this.redisClient.del(key)
    await this.redisClient.setRemove(getRedisBotSet(), key)
  }

  /**
   * Check if bot data corresponds to a real bot
   *
   * @param botData
   */
  public checkBotCreated(botData: BotData): boolean {
    return botData.botId !== ""
  }

  /**
   * Save the data for a bot, marking it as registered
   *
   * @param botData
   */
  private async saveBot(botData: BotData): Promise<void> {
    const key = getRedisBotKey(botData)
    const value = JSON.stringify(botData)
    await this.redisClient.set(key, value)
    await this.redisClient.setAdd(getRedisBotSet(), key)
  }

  /**
   * Create a new bot user
   *
   * @param botData
   */
  private makeBot(botData: BotData): Bot {
    Logger.info(
      `Creating bot for project ${botData.projectName}, task ${botData.taskIndex}`
    )
    const bot = new Bot(
      botData,
      this.config.bot.host,
      this.config.bot.port,
      this.config.redis
    )

    // Only use this disable if we are certain all the errors are handled
    // eslint-disable-next-line @typescript-eslint/no-misused-promises
    const pollId = setInterval(async () => {
      await this.monitorActivity(bot, pollId)
    }, this.pollTime)
    return bot
  }

  /**
   * Kill the bot if no activity since time of last poll
   *
   * @param bot
   * @param pollId
   */
  private async monitorActivity(
    bot: Bot,
    pollId: NodeJS.Timeout
  ): Promise<void> {
    if (bot.getActionCount() > 0) {
      bot.resetActionCount()
    } else {
      clearInterval(pollId)
      await this.deleteBot(bot.getData())
      bot.setActivate(false)
    }
  }
}
