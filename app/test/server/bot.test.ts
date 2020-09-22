import axios from "axios"
import io from "socket.io-client"

import { configureStore } from "../../src/common/configure_store"
import { uid } from "../../src/common/uid"
import { index2str } from "../../src/common/util"
import { EventName } from "../../src/const/connection"
import { Bot } from "../../src/server/bot"
import { serverConfig } from "../../src/server/defaults"
import { AddLabelsAction } from "../../src/types/action"
import { ItemExport } from "../../src/types/export"
import {
  ActionPacketType,
  BotData,
  RegisterMessageType,
  SyncActionMessageType
} from "../../src/types/message"
import { ReduxStore } from "../../src/types/redux"
import { State } from "../../src/types/state"
import {
  getInitialState,
  getRandomBox2dAction,
  getRandomModelPoly
} from "./util/util"

/**
 * Mock post request to model server
 * They should return the same number of prediction actions as request actions
 */
jest.mock("axios")
axios.post = jest
  .fn()
  .mockImplementation((_endpoint: string, data: ItemExport[]) => {
    const points = []
    for (const _d of data) {
      points.push(getRandomModelPoly())
    }
    return {
      status: 200,
      data: {
        points
      }
    }
  })

let botData: BotData
const socketEmit = jest.fn()
const mockSocket = {
  on: jest.fn(),
  connected: true,
  emit: socketEmit
}
let host: string
let port: number
let webId: string
let projectName: string
let initialState: State

beforeAll(() => {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  projectName = "testProject"
  botData = {
    taskIndex: 0,
    projectName,
    botId: "fakeBotId",
    address: location.origin
  }
  host = serverConfig.bot.host
  port = serverConfig.bot.port
  webId = "fakeUserId"
  initialState = getInitialState(webId)
})

// Note that these tests are similar to the frontend tests for synchronizer
describe("Test simple bot functionality", () => {
  test("Test data access", async () => {
    const bot = new Bot(botData, host, port)
    expect(bot.getData()).toEqual(botData)
  })

  test("Test correct registration message gets sent", async () => {
    const bot = new Bot(botData, host, port)
    bot.connectHandler()

    checkConnectMessage(bot.sessionId)
  })
})

describe("Test bot send-ack loop", () => {
  test("Test single packet prediction", async () => {
    const bot = setUpBot()
    const numActions = 5
    const message = makeSyncMessage(numActions, webId)

    // Check initial count
    expect(bot.getActionCount()).toBe(0)

    // Send the action packet
    const botActions = await bot.actionBroadcastHandler(message)
    expect(bot.getActionCount()).toBe(numActions)
    expect(botActions.length).toBe(numActions)

    // Verify that the trigger id is set correctly
    const calls = socketEmit.mock.calls
    const args = calls[calls.length - 1]
    expect(args[0]).toBe(EventName.ACTION_SEND)
    expect(args[1].actions.triggerId).toBe(message.actions.id)
  })

  test("Test duplicate actions are ignored", async () => {
    const bot = setUpBot()
    const numActions = 5
    const message = makeSyncMessage(numActions, webId)

    // Check initial count
    expect(bot.getActionCount()).toBe(0)

    // Send the action packet
    let botActions = await bot.actionBroadcastHandler(message)
    expect(bot.getActionCount()).toBe(numActions)
    expect(botActions.length).toBe(numActions)

    // Send the duplicate packet
    botActions = await bot.actionBroadcastHandler(message)
    expect(bot.getActionCount()).toBe(numActions)
    expect(botActions.length).toBe(0)
  })

  test("Test bot actions are ignored", async () => {
    const bot = setUpBot()
    const numActions = 5
    const botMessage = makeSyncMessage(numActions, bot.sessionId)

    // Check initial count
    expect(bot.getActionCount()).toBe(0)

    // Send the bot action packet
    const botActions = await bot.actionBroadcastHandler(botMessage)
    expect(bot.getActionCount()).toBe(0)
    expect(botActions.length).toBe(0)
  })

  test("Test bot store updates correctly", async () => {
    const bot = setUpBot()
    const expectedStore = configureStore(initialState)

    // Check initial count
    expect(bot.getActionCount()).toBe(0)

    // Make the messages
    const numMessages = 5
    const messages = []
    const actionsPerMessage = []
    for (let _ = 0; _ < numMessages; _++) {
      // Random int from 1 to 10
      const numActions = 1 + Math.floor(Math.random() * 10)
      actionsPerMessage.push(numActions)
      messages.push(makeSyncMessage(numActions, webId))
    }

    let totalActions = 0
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i]
      const numActions = actionsPerMessage[i]
      const botActions = await bot.actionBroadcastHandler(message)
      updateExpectedStore(expectedStore, message, botActions)

      totalActions += numActions
      expect(bot.getActionCount()).toBe(totalActions)
      expect(botActions.length).toBe(numActions)
    }

    // Check the final store
    expect(expectedStore.getState().present).toStrictEqual(bot.getState())

    // Reset the action count
    bot.resetActionCount()
    expect(bot.getActionCount()).toBe(0)
  })
})

/**
 * Creates the bot and initializes its store using the register handler
 */
function setUpBot(): Bot {
  const bot = new Bot(botData, host, port)
  bot.registerAckHandler(initialState)
  return bot
}

/**
 * Helper function to update the expected store with
 * the incoming actions and the outgoing predictions
 *
 * @param store
 * @param message
 * @param botActions
 */
function updateExpectedStore(
  store: ReduxStore,
  message: SyncActionMessageType,
  botActions: AddLabelsAction[]
): void {
  // Apply incoming actions
  for (const action of message.actions.actions) {
    store.dispatch(action)
  }
  // Apply outgoing predictions
  for (const botAction of botActions) {
    store.dispatch(botAction)
  }
}
/**
 * Helper function for checking that correct connection message was sent
 *
 * @param sessId
 */
function checkConnectMessage(sessId: string): void {
  const expectedMessage: RegisterMessageType = {
    projectName: botData.projectName,
    taskIndex: botData.taskIndex,
    sessionId: sessId,
    userId: botData.botId,
    address: location.origin,
    bot: true
  }
  expect(socketEmit).toHaveBeenCalledWith(EventName.REGISTER, expectedMessage)
}

/**
 * Create a sync message with the specified number of actions
 *
 * @param numActions
 * @param userId
 */
function makeSyncMessage(
  numActions: number,
  userId: string
): SyncActionMessageType {
  const actions: AddLabelsAction[] = []
  for (let _ = 0; _ < numActions; _++) {
    actions.push(getRandomBox2dAction())
  }
  const packet: ActionPacketType = {
    actions,
    id: uid()
  }
  return packetToMessage(packet, userId)
}

/**
 * Convert action packet to sync message
 *
 * @param packet
 * @param sessionId
 */
function packetToMessage(
  packet: ActionPacketType,
  sessionId: string
): SyncActionMessageType {
  return {
    actions: packet,
    projectName: botData.projectName,
    sessionId,
    taskId: index2str(botData.taskIndex),
    bot: false
  }
}
