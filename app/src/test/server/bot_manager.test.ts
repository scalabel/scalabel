import _ from 'lodash'
import { BotManager } from '../../js/server/bot_manager'
import { RedisClient } from '../../js/server/redis_client'
import { RedisPubSub } from '../../js/server/redis_pub_sub'
import { RegisterMessageType, ServerConfig } from '../../js/server/types'
import { sleep } from '../project/util'
import { getTestConfig } from '../util'

let client: RedisClient
let subClient: RedisClient
let subscriber: RedisPubSub
let config: ServerConfig
let projectName: string
let taskIndex: number
let userId: string
let sessionId: string
let address: string
let registerData: RegisterMessageType

beforeAll(async () => {
  config = getTestConfig()
  client = new RedisClient(config)
  subClient = new RedisClient(config)
  subscriber = new RedisPubSub(subClient)
  projectName = 'projectName'
  taskIndex = 0
  userId = 'userId'
  sessionId = 'sessionId'
  address = 'address'
  registerData = {
    projectName,
    taskIndex,
    sessionId,
    userId,
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
    expect(await botManager.checkBotExists(userId)).toBe(false)

    // register a new bot
    registerData.bot = false
    const message = JSON.stringify(registerData)
    const bot = await botManager.handleRegister('', message)

    // check that bot instance was created correctly with a single session
    expect(bot.webId).toBe(userId)
    expect(bot.address).toBe(address)
    expect(bot.sessions.length).toBe(1)
    expect(bot.botId).not.toBe('')

    // check that it was stored in redis
    expect(await botManager.checkBotExists(userId)).toBe(true)
    const botData = await botManager.getBot(userId)
    expect(botData).toEqual(bot.getData())

    // make sure only a dummy bot is created if you register again
    let dummyBot = await botManager.handleRegister('', message)
    expect(dummyBot.botId).toBe('')

    // make sure only a dummy bot is created if a bot registers
    registerData.bot = true
    registerData.userId = bot.botId
    const botMessage = JSON.stringify(registerData)
    dummyBot = await botManager.handleRegister('', botMessage)
    expect(dummyBot.botId).toBe('')

    // test that the bot is restored correctly
    const oldBots = await botManager.restoreUsers()
    expect(oldBots.length).toBe(1)
    expect(oldBots[0].getData()).toEqual(bot.getData())
  })

  test('Test deregistration after no activity', async () => {
    const msTimeout = 300
    const botManager = new BotManager(config, subscriber, client, msTimeout)
    const newUserId = 'newUserId'

    // make sure redis is empty initially
    expect(await botManager.checkBotExists(newUserId)).toBe(false)

    // register a new bot
    registerData.bot = false
    registerData.userId = newUserId
    const message = JSON.stringify(registerData)
    await botManager.handleRegister('', message)
    expect(await botManager.checkBotExists(newUserId)).toBe(true)

    // no actions occur, so after timeout, bot should be deleted
    await sleep(1000)
    expect(await botManager.checkBotExists(newUserId)).toBe(false)
  })
})
