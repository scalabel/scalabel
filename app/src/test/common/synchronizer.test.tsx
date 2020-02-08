import { cleanup } from '@testing-library/react'
import express, { Application } from 'express'
import { createServer } from 'http'
import socketio from 'socket.io'
import { Synchronizer } from '../../js/common/synchronizer'
import { makeState } from '../../js/functional/states'
import { EventName } from '../../js/server/types'

let io: socketio.Server

beforeAll(() => {
  // start http and socket io servers
  const app: Application = express()
  const httpServer = createServer(app)
  io = socketio(httpServer)

  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, (_rawData: string) => {
      const defaultState = makeState()
      socket.emit(EventName.REGISTER_ACK, defaultState)
    })
  })
})

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test sending until acked', () => {
    const taskIndex = 0
    const projectName = 'testProject'
    const userId = 'user'
    const callback = () => { return }
    const synchronizer = new Synchronizer(
      taskIndex, projectName, userId, callback)
    synchronizer.sendQueuedActions()
  })
})
