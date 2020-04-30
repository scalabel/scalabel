import axios from 'axios'
import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import { AddLabelsAction } from '../../js/action/types'
import { Bot } from '../../js/server/bot'
import { serverConfig } from '../../js/server/defaults'
import {
  ActionPacketType, BotData, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { index2str } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from './util/util'

jest.mock('axios')
axios.post = jest.fn().mockImplementation(() => {
  return {
    status: 200,
    data: {
      points:  [[[1, 2], [3, 4]]]
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

beforeAll(() => {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  botData = {
    taskIndex: 0,
    projectName: 'testProject',
    botId: 'fakeBotId',
    address: location.origin
  }
  host = serverConfig.botHost
  port = serverConfig.botPort
  webId = 'fakeUserId'
})

// Note that these tests are similar to the frontend tests for synchronizer
describe('Test bot functionality', () => {
  test('Test data access', async () => {
    const bot = new Bot(botData, host, port)
    expect(bot.getData()).toEqual(botData)
  })

  test('Test correct registration message gets sent', async () => {
    const bot = new Bot(botData, host, port)
    bot.connectHandler()

    checkConnectMessage(bot.sessionId)
  })

  test('Test send-ack loop', async () => {
    const bot = new Bot(botData, host, port)

    const message1 = makeSyncMessage(1, webId)
    const message2 = makeSyncMessage(2, webId)
    const message3 = makeSyncMessage(1, bot.sessionId)

    // Set up the store with register ack
    const initState = getInitialState(webId)
    bot.registerAckHandler(initState)

    // Check initial count
    expect(bot.getActionCount()).toBe(0)

    // Send single action message
    await bot.actionBroadcastHandler(message1)
    expect(bot.getActionCount()).toBe(1)

    // Verify that the trigger id is set correctly
    const calls = socketEmit.mock.calls
    const args = calls[calls.length - 1]
    expect(args[0]).toBe(EventName.ACTION_SEND)
    expect(args[1].actions.triggerId).toBe(message1.actions.id)

    // Duplicates should be ignored
    await bot.actionBroadcastHandler(message1)
    expect(bot.getActionCount()).toBe(1)

    // Send a 2-action message
    await bot.actionBroadcastHandler(message2)
    expect(bot.getActionCount()).toBe(3)

    // Bot messages should be ignored
    await bot.actionBroadcastHandler(message3)
    expect(bot.getActionCount()).toBe(3)

    // Reset the action count
    bot.resetActionCount()
    expect(bot.getActionCount()).toBe(0)
  })
})

/**
 * Helper function for checking that correct connection message was sent
 */
function checkConnectMessage (sessId: string) {
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
 */
function makeSyncMessage (
  numActions: number, userId: string): SyncActionMessageType {
  const actions: AddLabelsAction[] = []
  for (let _ = 0; _ < numActions; _++) {
    actions.push(getRandomBox2dAction())
  }
  const packet: ActionPacketType = {
    actions,
    id: uuid4()
  }
  return packetToMessage(packet, userId)
}

/**
 * Convert action packet to sync message
 */
function packetToMessage (
  packet: ActionPacketType, sessionId: string): SyncActionMessageType {
  return {
    actions: packet,
    projectName: botData.projectName,
    sessionId,
    taskId: index2str(botData.taskIndex),
    bot: false
  }
}
