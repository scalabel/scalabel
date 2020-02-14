import { cleanup } from '@testing-library/react'
import io from 'socket.io-client'
import { addBox2dLabel } from '../../js/action/box2d'
import { BaseAction } from '../../js/action/types'
import { configureStore } from '../../js/common/configure_store'
import Session, { ConnectionStatus } from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { makeItem,
  makeSensor, makeState, makeTask } from '../../js/functional/states'
import { State, TaskType } from '../../js/functional/types'
import { updateState } from '../../js/server/util'

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test send-ack loop', async () => {
    const sessionId = 'fakeSessId'
    const synchronizer = startSynchronizer(sessionId)
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
    synchronizer.actionBroadcastHandler(synchronizer.listActionPackets()[0])
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status).toBe(ConnectionStatus.NOTIFY_SAVED)
  })

  test('Test reconnection', async () => {
    Session.updateStatus(ConnectionStatus.UNSAVED)
    const sessionId = 'fakeSessId'
    const synchronizer = startSynchronizer(sessionId)
    const initialState = getInitialState(sessionId)
    synchronizer.registerAckHandler(initialState)

    // Initially, no actions are queued for saving
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status).toBe(ConnectionStatus.UNSAVED)

    // Dispatch an action to trigger a sync event
    const frontendAction = randomBox2dAction()
    Session.dispatch(frontendAction)

    // Before ack, dispatched action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status).toBe(ConnectionStatus.SAVING)

    // Backend disconnects instead of acking
    synchronizer.disconnectHandler()
    expect(Session.status).toBe(ConnectionStatus.RECONNECTING)

    // Reconnect, but some missed actions occured in the backend
    const missedAction = randomBox2dAction()
    const newInitialState = updateState(initialState, [missedAction], false)
    synchronizer.registerAckHandler(newInitialState)

    // Check that frontend state updates correctly
    const expectedState = updateState(newInitialState, [frontendAction], false)
    expect(Session.getState()).toMatchObject(expectedState)
    // Also check that save is still in progress
    checkNumQueuedActions(synchronizer, 1)
    checkFirstAction(synchronizer, frontendAction)
    expect(Session.status).toBe(ConnectionStatus.SAVING)

    // After ack arrives, no actions are queued anymore
    synchronizer.actionBroadcastHandler(synchronizer.listActionPackets()[0])
    expect(Session.getState()).toMatchObject(expectedState)
    checkNumQueuedActions(synchronizer, 0)
    expect(Session.status).toBe(ConnectionStatus.NOTIFY_SAVED)
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
function startSynchronizer (sessionId: string): Synchronizer {
  // start frontend synchronizer
  const taskIndex = 0
  const projectName = 'testProject'
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
 * The initial backend task represents the saved data
 */
function getInitialState (sessionId: string): State {
  const partialTask: Partial<TaskType> = {
    items: [makeItem({ id: 0 })],
    sensors: { 0: makeSensor(0, '', '') }
  }
  const defaultTask = makeTask(partialTask)
  const defaultState = makeState({
    task: defaultTask
  })
  defaultState.session.id = sessionId
  defaultState.task.config.autosave = true
  return defaultState
}

/**
 * Helper function to get box2d actions
 */
export function randomBox2dAction () {
  return addBox2dLabel(0, 0, [], {},
    Math.random(), Math.random(), Math.random(), Math.random())
}
