import axios from 'axios'
import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import { Bot } from '../../js/server/bot'
import { serverConfig } from '../../js/server/defaults'
import {
  ActionPacketType, BotData, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { index2str } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from '../util'

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

// Note- these tests are similar to the frontend tests for synchronizer
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

    const packet1: ActionPacketType = {
      actions: [getRandomBox2dAction()],
      id: uuid4()
    }
    const message1 = packetToMessage(packet1, webId)
    const packet2: ActionPacketType = {
      actions: [getRandomBox2dAction(), getRandomBox2dAction()],
      id: uuid4()
    }
    const message2 = packetToMessage(packet2, webId)
    const packet3: ActionPacketType = {
      actions: [getRandomBox2dAction()],
      id: uuid4()
    }
    const message3 = packetToMessage(packet3, bot.sessionId)

    // set up the store with register ack
    const initState = getInitialState(webId)
    bot.registerAckHandler(initState)

    // check initial count
    expect(bot.actionCount).toBe(0)

    // send single action message
    await bot.actionBroadcastHandler(message1)
    expect(bot.actionCount).toBe(1)

    // duplicates should be ignored
    await bot.actionBroadcastHandler(message1)
    expect(bot.actionCount).toBe(1)

    // send a 2-action message
    await bot.actionBroadcastHandler(message2)
    expect(bot.actionCount).toBe(3)

    // bot messages should be ignored
    await bot.actionBroadcastHandler(message3)
    expect(bot.actionCount).toBe(3)

    // and reset the action count
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
