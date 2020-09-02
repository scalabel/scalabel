import { BotManager } from "../../src/server/bot_manager"
import { getRedisBotKey } from "../../src/server/path"
import { RedisClient } from "../../src/server/redis_client"
import { RedisPubSub } from "../../src/server/redis_pub_sub"
import { ServerConfig } from "../../src/types/config"
import { BotData, RegisterMessageType } from "../../src/types/message"
import { sleep } from "../project/util"
import { getTestConfig } from "./util/util"

let client: RedisClient
let subClient: RedisClient
let subscriber: RedisPubSub
let config: ServerConfig

beforeAll(async () => {
  config = getTestConfig()
  client = new RedisClient(config.redis)
  subClient = new RedisClient(config.redis)
  subscriber = new RedisPubSub(subClient)
})

afterAll(async () => {
  await client.close()
  await subClient.close()
})

describe("Test bot user manager", () => {
  test("Test registration", async () => {
    const botManager = new BotManager(config.bot, subscriber, client)

    // Test that different tasks create different bots
    const goodRegisterMessages: RegisterMessageType[] = [
      makeRegisterData("project", 0, "user", false),
      makeRegisterData("project", 1, "user", false),
      makeRegisterData("projectOther", 0, "user", false),
      makeRegisterData("projectOther", 2, "user2", false)
    ]

    for (const registerData of goodRegisterMessages) {
      // Make sure redis is empty initially
      const dummyBotData = makeBotData(registerData, "botId")
      expect(await botManager.checkBotExists(dummyBotData)).toBe(false)

      // Register a new bot
      const botData = await botManager.handleRegister(
        "",
        JSON.stringify(registerData)
      )

      // Should match register data, and generate an id
      expect(botData.projectName).toBe(registerData.projectName)
      expect(botData.taskIndex).toBe(registerData.taskIndex)
      expect(botData.address).toBe(registerData.address)
      expect(botManager.checkBotCreated(botData)).toBe(true)

      // Check that it was stored in redis
      expect(await botManager.checkBotExists(botData)).toBe(true)
      const redisBotData = await botManager.getBot(getRedisBotKey(botData))
      expect(redisBotData).toEqual(botData)
    }

    // Make sure only dummy bots are created for the following cases:
    const badRegisterMessages: RegisterMessageType[] = [
      // Duplicated messages
      goodRegisterMessages[0],
      // Same task, different user
      makeRegisterData("project", 0, "user2", false),
      // Bot user
      makeRegisterData("project", 0, "user", true)
    ]
    for (const registerData of badRegisterMessages) {
      const fakeBotData = await botManager.handleRegister(
        "",
        JSON.stringify(registerData)
      )
      expect(botManager.checkBotCreated(fakeBotData)).toBe(false)
    }

    // Test that the bots are restored correctly
    const oldBots = await botManager.restoreBots()
    expect(oldBots.length).toBe(goodRegisterMessages.length)
  })

  test("Test deregistration after no activity", async () => {
    const msTimeout = 300
    const botManager = new BotManager(config.bot, subscriber, client, msTimeout)
    const registerData = makeRegisterData("project2", 0, "user2", false)
    const botData = makeBotData(registerData, "botId")

    // Make sure redis is empty initially
    expect(await botManager.checkBotExists(botData)).toBe(false)

    const message = JSON.stringify(registerData)
    await botManager.handleRegister("", message)
    expect(await botManager.checkBotExists(botData)).toBe(true)

    // No actions occur, so after timeout, bot should be deleted
    await sleep(1000)
    expect(await botManager.checkBotExists(botData)).toBe(false)
  })
})

/**
 * Create data for registration with some defaults
 *
 * @param projectName
 * @param taskIndex
 * @param userId
 * @param bot
 */
function makeRegisterData(
  projectName: string,
  taskIndex: number,
  userId: string,
  bot: boolean
): RegisterMessageType {
  return {
    projectName,
    taskIndex,
    sessionId: "sessionId",
    userId,
    address: "address",
    bot
  }
}

/**
 * Create default data for a bot
 *
 * @param registerData
 * @param botId
 */
function makeBotData(
  registerData: RegisterMessageType,
  botId: string
): BotData {
  return {
    projectName: registerData.projectName,
    taskIndex: registerData.taskIndex,
    address: registerData.address,
    botId
  }
}
