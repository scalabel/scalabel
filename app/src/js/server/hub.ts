import * as fs from 'fs-extra'
import { Store } from 'redux'
import * as socketio from 'socket.io'
import { sprintf } from 'sprintf-js'
import * as uuid from 'uuid'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { State } from '../functional/types'
import * as path from './path'
import Session from './server_session'
import { ErrorCode, EventName } from './types'

/* TODO: convert these handlers to use the new backend functionality
 * including global storage
 */
/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (io: socketio.Server) {
  const env = Session.getEnv()
  // maintain a store for each task
  const stores: { [key: string]: Store } = {}
  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, (state: State) => {
      const taskId = state.task.config.taskId
      const projectName = state.task.config.projectName

      // TODO: add session id to state (instead of getting it from go backend)
      const sessId = uuid()

      const savePath = path.getPath(env.data, projectName, taskId)
      const room = path.roomName(projectName, taskId, sessId, env.sync)

      // If there's no store in memory, try to load it
      if (!stores[room]) {
        /* frontend data is used to initialize a valid store
          but the non-task data is arbitrary */
        try {
          // Try to load task state from file system
          const fileNames = fs.readdirSync(savePath)
          // Get most recently saved file
          const fileName = fileNames[fileNames.length - 1]
          const file = sprintf('%s/%s', savePath, fileName)
          const rawContent = fs.readFileSync(file, 'utf8')
          const content = JSON.parse(rawContent)
          // Combine loaded task state with other state sent from frontend
          state.task = content
        } catch (e) {
          /* Don't crash if file does not exist; this just means it has not
            been initialized yet */
          if (e.code !== ErrorCode.NO_FILE) {
            throw(e)
          }
        }
        stores[room] = configureStore(state)
        // automatically save task data on updates
        stores[room].subscribe(() => {
          const content = JSON.stringify(stores[room].getState().present.task)
          const file = path.getFile(savePath)
          fs.writeFileSync(file, content)
        })
      } else {
        state.task = stores[room].getState().present.task
      }

      // Connect socket to others in the same room
      socket.join(room)
      // Send backend state to newly registered socket
      socket.emit(EventName.REGISTER_ACK, state)
    })

    socket.on(EventName.ACTION_SEND, (rawData: string) => {
      const data = JSON.parse(rawData)
      const projectName = data.project
      const taskId = data.taskId
      const sessId = data.sessId
      const actionList = data.actions

      const savePath = path.getPath(env.data, projectName, taskId)
      const room = path.roomName(projectName, taskId, sessId, env.sync)

      // Make sure file exists before saving to it
      fs.ensureDirSync(savePath)
      // For each action, update the backend store and broadcast
      for (const action of actionList) {
        action.timestamp = Date.now()
        // for task actions, update store and broadcast to room
        if (types.TASK_ACTION_TYPES.includes(action.type)) {
          stores[room].dispatch(action)
          io.in(room).emit(EventName.ACTION_BROADCAST, action)
        } else {
          socket.emit(EventName.ACTION_BROADCAST, action)
        }
      }
    })
  })
}
