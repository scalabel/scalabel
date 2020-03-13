import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import { Bot } from '../../js/server/bot'
import {
  ActionPacketType, BotData, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { index2str } from '../../js/server/util'
import { getRandomBox2dAction } from '../util'

let botData: BotData
const socketEmit = jest.fn()
const mockSocket = {
  on: jest.fn(),
  connected: true,
  emit: socketEmit
}

beforeAll(() => {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  botData = {
    taskIndex: 0,
    projectName: 'testProject',
    botId: 'fakeUserId',
    address: location.origin
  }
})

// Note- these tests are similar to the frontend tests for synchronizer
describe('Test bot functionality', () => {
  test('Test data access', async () => {
    const bot = new Bot(botData)
    expect(bot.getData()).toEqual(botData)
  })

  test('Test correct registration message gets sent', async () => {
    const bot = new Bot(botData)
    bot.connectHandler()

    checkConnectMessage(bot.sessionId)
  })

  test('Test send-ack loop', async () => {
    const bot = new Bot(botData)

    const packet1: ActionPacketType = {
      actions: [getRandomBox2dAction()],
      id: uuid4()
    }
    const message1 = packetToMessage(packet1, bot.sessionId)
    const packet2: ActionPacketType = {
      actions: [getRandomBox2dAction(), getRandomBox2dAction()],
      id: uuid4()
    }
    const message2 = packetToMessage(packet2, bot.sessionId)

    // check initial count
    expect(bot.actionCount).toBe(0)

    // send single action message
    bot.actionBroadcastHandler(message1)
    expect(bot.actionCount).toBe(1)

    // duplicates should be ignored
    bot.actionBroadcastHandler(message1)
    expect(bot.actionCount).toBe(1)

    // finally send a 2-action message
    bot.actionBroadcastHandler(message2)
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
    taskId: index2str(botData.taskIndex)
  }
}
