import _ from 'lodash'
import { BotManager } from '../../js/server/bot_manager'
import { getRedisBotKey } from '../../js/server/path'
import { RedisClient } from '../../js/server/redis_client'
import { RedisPubSub } from '../../js/server/redis_pub_sub'
import {
  BotData, RegisterMessageType, ServerConfig } from '../../js/server/types'
import { sleep } from '../project/util'
import { getTestConfig } from '../util'

let client: RedisClient
let subClient: RedisClient
let subscriber: RedisPubSub
let config: ServerConfig
let sessionId: string
let botData: BotData
let registerData: RegisterMessageType

beforeAll(async () => {
  config = getTestConfig()
  client = new RedisClient(config)
  subClient = new RedisClient(config)
  subscriber = new RedisPubSub(subClient)
  const projectName = 'projectName'
  const taskIndex = 0
  const address = 'address'
  botData = {
    projectName,
    taskIndex,
    botId: 'botId',
    address
  }
  sessionId = 'sessionId'

  registerData = {
    projectName,
    taskIndex,
    sessionId,
    userId: 'userId',
    address,
    bot: false
  }
})

afterAll(async () => {
  await subClient.close()
})

describe('Test bot user manager', () => {
  test('Test registration', async () => {
    const botManager = new BotManager(config, subscriber, client)

    // make sure redis is empty initially
    expect(await botManager.checkBotExists(botData)).toBe(false)

    // register a new bot
    registerData.bot = false
    const message = JSON.stringify(registerData)
    const bot = await botManager.handleRegister('', message)

    // should match register data, and generate an id
    expect(bot.projectName).toBe(registerData.projectName)
    expect(bot.taskIndex).toBe(registerData.taskIndex)
    expect(bot.address).toBe(registerData.address)
    expect(bot.botId).not.toBe('')

    // check that it was stored in redis
    expect(await botManager.checkBotExists(botData)).toBe(true)
    const redisBotData = await botManager.getBot(getRedisBotKey(botData))
    expect(redisBotData).toEqual(bot)

    // make sure only a dummy bot is created if you register again
    let dummyBot = await botManager.handleRegister('', message)
    expect(dummyBot.botId).toBe('')

    // make sure only a dummy bot is created if a bot registers
    const newRegisterData = {
      ...registerData,
      bot: true,
      userId: bot.botId
    }
    const botMessage = JSON.stringify(newRegisterData)
    dummyBot = await botManager.handleRegister('', botMessage)
    expect(dummyBot.botId).toBe('')

    // test that the bot is restored correctly
    const oldBots = await botManager.restoreBots()
    expect(oldBots.length).toBe(1)
    expect(oldBots[0].getData()).toEqual(bot)
  })

  test('Test deregistration after no activity', async () => {
    const msTimeout = 300
    const botManager = new BotManager(config, subscriber, client, msTimeout)
    const projectName = 'newProject'
    const newBotData = {
      ...botData,
      projectName
    }

    // make sure redis is empty initially
    expect(await botManager.checkBotExists(newBotData)).toBe(false)

    // register a new bot
    const newRegisterData = {
      ...registerData,
      projectName
    }
    const message = JSON.stringify(newRegisterData)
    await botManager.handleRegister('', message)
    expect(await botManager.checkBotExists(newBotData)).toBe(true)

    // no actions occur, so after timeout, bot should be deleted
    await sleep(1000)
    expect(await botManager.checkBotExists(newBotData)).toBe(false)
  })
})
