import { cleanup } from '@testing-library/react'
import express, { Application } from 'express'
import { createServer, Server } from 'http'
import socketio from 'socket.io'
import { BaseAction } from '../../js/action/types'
import { configureStore } from '../../js/common/configure_store'
import Session from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { makeItem,
  makeSensor, makeState, makeTask } from '../../js/functional/states'
import { State, TaskType } from '../../js/functional/types'
import { EventName, SyncActionMessageType } from '../../js/server/types'
import { sleep } from '../project/util'

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test send-ack loop', async () => {
    const sessionId = 'fakeSessId'
    const server = startServer(0, sessionId)
    const synchronizer = startSynchronizer(sessionId)
    // wait for registration
    await sleep(100)

    const dummyAction: BaseAction = {
      type: 'a',
      sessionId
    }
    // no actions queued for saving
    checkNumQueuedActions(synchronizer, 0)
    // dispatching an action triggers a sync event
    Session.dispatch(dummyAction)
    await sleep(100)
    // ack arrives, so no actions queued anymore
    checkNumQueuedActions(synchronizer, 0)

    killServer(server)
    await sleep(50)
  })

  test('Test send-ack loop with delay', async () => {
    const sessionId = 'fakeSessId'
    // add a delay of 100 ms
    const backendDelay = 100
    const server = startServer(backendDelay, sessionId)
    const synchronizer = startSynchronizer(sessionId)
    // wait for registration
    await sleep(200)

    const dummyAction: BaseAction = {
      type: 'a',
      sessionId
    }

    // no actions queued for saving
    checkNumQueuedActions(synchronizer, 0)
    // dispatching an action triggers a sync event
    Session.dispatch(dummyAction)
    await sleep(backendDelay / 2)

    // after only half of delay, action is still queued for saving
    checkNumQueuedActions(synchronizer, 1)
    await sleep(backendDelay / 2 + 50)

    // after full delay, ack arrives so no action queued anymore
    checkNumQueuedActions(synchronizer, 0)

    killServer(server)
    await sleep(50)
  })
})

/**
 * Helper function for checking the number of actions waiting to be saved
 */
function checkNumQueuedActions (sync: Synchronizer, num: number) {
  expect(Object.keys(sync.actionsToSave).length).toBe(num)
}

/**
 * Start the synchronizer being tested
 */
function startSynchronizer (sessionId: string): Synchronizer {
  // start frontend synchronizer
  const taskIndex = 0
  const projectName = 'testProject'
  const userId = 'user'
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
 * Start the mock server
 */
function startServer (backendDelay: number, sessionId: string): Server {
  // start http and socket io servers
  const app: Application = express()
  const httpServer: Server = createServer(app)
  const io: socketio.Server = socketio(httpServer)

  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, (_rawData: string) => {
      socket.emit(EventName.REGISTER_ACK, getInitialState(sessionId))
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      const data: SyncActionMessageType = JSON.parse(rawData)
      await sleep(backendDelay)
      socket.emit(EventName.ACTION_BROADCAST, data.actions)
    })
  })

  httpServer.listen(8687)

  return httpServer
}

/**
 * Kill the mock server
 */
function killServer (server: Server) {
  server.close()
  server.emit('close')
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
