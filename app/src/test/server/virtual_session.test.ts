import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import {
  ActionPacketType, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { index2str } from '../../js/server/util'
import { VirtualSession } from '../../js/server/virtual_session'
import { getRandomBox2dAction } from '../util'

let taskIndex: number
let projectName: string
let userId: string
const socketEmit = jest.fn()
const mockSocket = {
  on: jest.fn(),
  connected: true,
  emit: socketEmit
}

beforeAll(() => {
  taskIndex = 0
  projectName = 'testProject'
  userId = 'fakeUserId'
})

// Note- these tests are similar to the frontend tests for synchronizer
describe('Test virtual session functionality', () => {
  test('Test correct registration message gets sent', async () => {
    const sess = startSession()
    sess.connectHandler()

    checkConnectMessage(sess.sessionId)
  })

  test('Test send-ack loop', async () => {
    const sess = startSession()

    const packet1: ActionPacketType = {
      actions: [getRandomBox2dAction()],
      id: uuid4()
    }
    const message1 = packetToMessage(packet1, sess.sessionId)
    const packet2: ActionPacketType = {
      actions: [getRandomBox2dAction(), getRandomBox2dAction()],
      id: uuid4()
    }
    const message2 = packetToMessage(packet2, sess.sessionId)

    // check initial count
    expect(sess.actionCount).toBe(0)

    // send single action message
    sess.actionBroadcastHandler(message1)
    expect(sess.actionCount).toBe(1)

    // duplicates should be ignored
    sess.actionBroadcastHandler(message1)
    expect(sess.actionCount).toBe(1)

    // finally send a 2-action message
    sess.actionBroadcastHandler(message2)
    expect(sess.actionCount).toBe(3)
  })
})

/**
 * Helper function for checking that correct connection message was sent
 */
function checkConnectMessage (sessId: string) {
  const expectedMessage: RegisterMessageType = {
    projectName,
    taskIndex,
    sessionId: sessId,
    userId,
    address: location.origin,
    bot: true
  }
  expect(socketEmit).toHaveBeenCalledWith(EventName.REGISTER, expectedMessage)
}

/**
 * Start the virtual session instance
 */
function startSession (): VirtualSession {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  const synchronizer = new VirtualSession(
    userId,
    location.origin,
    projectName,
    taskIndex
  )

  return synchronizer
}

/**
 * Convert action packet to sync message
 */
function packetToMessage (
  packet: ActionPacketType, sessionId: string): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId,
    taskId: index2str(taskIndex)
  }
}
