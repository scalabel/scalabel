import { cleanup } from '@testing-library/react'
import io from 'socket.io-client'
import { BaseAction } from '../../js/action/types'
import { configureStore } from '../../js/common/configure_store'
import Session from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { State } from '../../js/functional/types'
import {
  ActionPacketType, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { index2str, updateState } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from '../util'

let sessionId: string
let botSessionId: string
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
  sessionId = 'fakeSessId'
  botSessionId = 'botSessId'
  taskIndex = 0
  projectName = 'testProject'
  userId = 'fakeUserId'
})

beforeEach(() => {
  Session.bots = false
  Session.status.setAsUnsaved()
})

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test correct registration message gets sent', async () => {
    // Since this deals with registration, don't initialize the state
    const initializeState = false
    const synchronizer = startSynchronizer(initializeState)
    synchronizer.connectHandler()

    // Frontend doesn't have a session id until after registration
    const expectedSessId = ''
    checkConnectMessage(expectedSessId)
    expect(Session.status.checkUnsaved()).toBe(true)
  })

  test('Test send-ack loop', async () => {
    const synchronizer = startSynchronizer()

    const dummyAction: BaseAction = {
      type: 'a',
      sessionId
    }

    // Initially, no actions queued for saving
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status.checkUnsaved()).toBe(true)

    // Dispatch an action to trigger a sync event
    Session.dispatch(dummyAction)

    // Before ack, dispatched action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, dummyAction)
    expect(Session.status.checkSaving()).toBe(true)

    // After ack arrives, no actions are queued anymore
    const ackAction = synchronizer.listActionPackets()[0]
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction))
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status.checkSaved()).toBe(true)
    checkNumLoggedActions(synchronizer, 1)

    // If second ack arrives, it is ignored
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction))
    checkNumLoggedActions(synchronizer, 1)
  })

  test('Test model prediction status', async () => {
    const synchronizer = startSynchronizer()
    Session.bots = true

    // Dispatch an add label action
    const boxAction = getRandomBox2dAction()
    Session.dispatch(boxAction)
    checkNumActionsPendingPrediction(synchronizer, 1)

    // After ack arrives, session status is marked as computing
    const ackAction = synchronizer.listActionPackets()[0]
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction))
    expect(Session.status.checkComputing()).toBe(true)

    // When model action arrives, it marks computation as finished
    const modelAction = getRandomBox2dAction()
    modelAction.sessionId = botSessionId

    const modelPacket: ActionPacketType = {
      actions: [modelAction],
      id: 'randomId',
      triggerId: ackAction.id
    }
    synchronizer.actionBroadcastHandler(
      packetToMessageBot(modelPacket))
    checkNumActionsPendingPrediction(synchronizer, 0)
    expect(Session.status.checkComputeDone()).toBe(true)
  })

  test('Test reconnection', async () => {
    const synchronizer = startSynchronizer()

    // Initially, no actions are queued for saving
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status.checkUnsaved()).toBe(true)

    // Dispatch an action to trigger a sync event
    const frontendAction = getRandomBox2dAction()
    Session.dispatch(frontendAction)

    // Before ack, dispatched action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status.checkSaving()).toBe(true)

    // Backend disconnects instead of acking
    synchronizer.disconnectHandler()
    expect(Session.status.checkReconnecting()).toBe(true)

    // Reconnect, but some missed actions occured in the backend
    const newInitialState = updateState(
      getInitialState(sessionId),
      [getRandomBox2dAction()]
    )
    synchronizer.connectHandler()
    checkConnectMessage(sessionId)
    synchronizer.registerAckHandler(newInitialState)

    // Check that frontend state updates correctly
    const expectedState = updateState(newInitialState, [frontendAction])
    expect(Session.getState()).toMatchObject(expectedState)
    // Also check that save is still in progress
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status.checkSaving()).toBe(true)

    // After ack arrives, no actions are queued anymore
    const ackAction = synchronizer.listActionPackets()[0]
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction))
    expect(Session.getState()).toMatchObject(expectedState)
    checkNumQueuedActions(synchronizer, 0)
  })
})

/**
 * Helper function for checking the number of actions waiting to be saved
 */
function checkNumQueuedActions (sync: Synchronizer, num: number) {
  const actionPackets = sync.listActionPackets()
  expect(actionPackets.length).toBe(num)
}

/**
 * Helper functions for checking the number of actions logged,
 * Which means they are confirmed as saved
 */
function checkNumLoggedActions (sync: Synchronizer, num: number) {
  expect(sync.actionLog.length).toBe(num)
}

/**
 * Helper functions for checking the number of add label actions
 * that are awaiting a response from a model
 */
function checkNumActionsPendingPrediction (sync: Synchronizer, num: number) {
  expect(sync.actionsPendingPrediction.size).toBe(num)

}

/**
 * Helper function for checking when one action should be queued
 */
function checkFirstAction (sync: Synchronizer, action: BaseAction) {
  const actionPackets = sync.listActionPackets()
  const actions = actionPackets[0].actions
  expect(actions.length).toBe(1)
  expect(actions[0]).toBe(action)
}

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
    bot: false
  }
  expect(socketEmit).toHaveBeenCalledWith(EventName.REGISTER, expectedMessage)
}

/**
 * Start the browser synchronizer being tested
 */
function startSynchronizer (setInitialState: boolean = true): Synchronizer {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  const synchronizer = new Synchronizer(
    taskIndex,
    projectName,
    userId,
    (backendState: State) => {
      backendState.session.id = sessionId
      backendState.task.config.autosave = true
      Session.store = configureStore(
        backendState, Session.devMode, synchronizer.middleware)
      Session.autosave = true
    }
  )

  if (setInitialState) {
    const initialState = getInitialState(sessionId)
    synchronizer.registerAckHandler(initialState)
  }

  return synchronizer
}

/**
 * Convert action packet to sync message
 */
function packetToMessage (packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId,
    taskId: index2str(taskIndex),
    bot: false
  }
}

/**
 * Convert action packet to sync message from a bot
 */
function packetToMessageBot (packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId: botSessionId,
    taskId: index2str(taskIndex),
    bot: true
  }
}
