import { cleanup } from '@testing-library/react'
import io from 'socket.io-client'
import { BaseAction } from '../../js/action/types'
import { configureStore } from '../../js/common/configure_store'
import Session, { ConnectionStatus } from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { State } from '../../js/functional/types'
import { ActionPacketType, SyncActionMessageType } from '../../js/server/types'
import { index2str, updateState } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from '../util'

let sessionId: string
let taskIndex: number
let projectName: string

beforeAll(() => {
  sessionId = 'fakeSessId'
  taskIndex = 0
  projectName = 'testProject'
})

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test send-ack loop', async () => {
    const synchronizer = startSynchronizer()
    const initialState = getInitialState(sessionId)
    synchronizer.registerAckHandler(initialState)

    const dummyAction: BaseAction = {
      type: 'a',
      sessionId
    }

    // Initially, no actions queued for saving
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status).toBe(ConnectionStatus.UNSAVED)

    // Dispatch an action to trigger a sync event
    Session.dispatch(dummyAction)

    // Before ack, dispatched action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, dummyAction)
    expect(Session.status).toBe(ConnectionStatus.SAVING)

    // After ack arrives, no actions are queued anymore
    const ackHandler = jest.fn()
    const ackAction = synchronizer.listActionPackets()[0]
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction), ackHandler)
    checkNumQueuedActions(synchronizer, 0)
    expect(ackHandler).toHaveBeenCalled()

    // If second ack arrives, it is ignored
    const newAckHandler = jest.fn()
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction), newAckHandler)
    expect(newAckHandler).not.toHaveBeenCalled()
  })

  test('Test reconnection', async () => {
    Session.updateStatus(ConnectionStatus.UNSAVED)
    const synchronizer = startSynchronizer()
    const initialState = getInitialState(sessionId)
    synchronizer.registerAckHandler(initialState)

    // Initially, no actions are queued for saving
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status).toBe(ConnectionStatus.UNSAVED)

    // Dispatch an action to trigger a sync event
    const frontendAction = getRandomBox2dAction()
    Session.dispatch(frontendAction)

    // Before ack, dispatched action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status).toBe(ConnectionStatus.SAVING)

    // Backend disconnects instead of acking
    synchronizer.disconnectHandler()
    expect(Session.status).toBe(ConnectionStatus.RECONNECTING)

    // Reconnect, but some missed actions occured in the backend
    const missedAction = getRandomBox2dAction()
    const newInitialState = updateState(initialState, [missedAction])
    synchronizer.registerAckHandler(newInitialState)

    // Check that frontend state updates correctly
    const expectedState = updateState(newInitialState, [frontendAction])
    expect(Session.getState()).toMatchObject(expectedState)
    // Also check that save is still in progress
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status).toBe(ConnectionStatus.SAVING)

    // After ack arrives, no actions are queued anymore
    const ackHandler = jest.fn()
    const ackAction = synchronizer.listActionPackets()[0]
    synchronizer.actionBroadcastHandler(
      packetToMessage(ackAction), ackHandler)
    expect(Session.getState()).toMatchObject(expectedState)
    checkNumQueuedActions(synchronizer, 0)
    expect(ackHandler).toHaveBeenCalled()
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
 * Helper function for checking when one action should be queued
 */
function checkFirstAction (sync: Synchronizer, action: BaseAction) {
  const actionPackets = sync.listActionPackets()
  const actions = actionPackets[0].actions
  expect(actions.length).toBe(1)
  expect(actions[0]).toBe(action)
}

/**
 * Start the synchronizer being tested
 */
function startSynchronizer (): Synchronizer {
  // start frontend synchronizer
  const userId = 'user'
  const mockSocket = {
    on: jest.fn(),
    connected: true,
    emit: jest.fn()
  }
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
    taskId: index2str(taskIndex)
  }
}
