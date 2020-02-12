import { cleanup } from '@testing-library/react'
import io from 'socket.io-client'
import { BaseAction } from '../../js/action/types'
import { configureStore } from '../../js/common/configure_store'
import Session from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { makeItem,
  makeSensor, makeState, makeTask } from '../../js/functional/states'
import { State, TaskType } from '../../js/functional/types'

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
    // no actions queued for saving
    checkNumQueuedActions(synchronizer, 0)
    // dispatching an action triggers a sync event
    Session.dispatch(dummyAction)

    // before ack, action is queued for saving
    checkNumQueuedActions(synchronizer, 1)
    const actionPackets = synchronizer.listActionPackets()
    const actions = actionPackets[0].actions
    expect(actions.length).toBe(1)
    expect(actions[0]).toBe(dummyAction)

    synchronizer.actionBroadcastHandler(actionPackets[0])

    // ack arrives, so no actions queued anymore
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
 * Start the synchronizer being tested
 */
function startSynchronizer (sessionId: string): Synchronizer {
  // start frontend synchronizer
  const taskIndex = 0
  const projectName = 'testProject'
  const userId = 'user'
  const mockSocket = io()
  mockSocket.connected = true
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
    },
    mockSocket
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
