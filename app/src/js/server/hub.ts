import express from 'express'
import * as fs from 'fs-extra'
import { createServer } from 'http'
import * as yaml from 'js-yaml'
import { Store } from 'redux'
import * as socketio from 'socket.io'
import { sprintf } from 'sprintf-js'
import * as yargs from 'yargs'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { State } from '../functional/types'
import * as path from './path'

export const enum EventName {
  ACTION_BROADCAST = 'actionBroadcast',
  ACTION_SEND = 'actionSend',
  REGISTER_ACK = 'registerAck',
  REGISTER = 'register',
  CONNECTION = 'connection',
  CONNECT = 'connect',
  DISCONNECT = 'disconnect'
}

const NO_FILE_ERROR_CODE = 'ENOENT'

const argv = yargs
  .option('config', {
    describe: 'Config file path.'
  })
  .demandOption('config')
  .string('config')
  .argv

const configDir: string = argv.config

// load the config for port info
let syncPort: number | undefined
let dataDir: string
try {
  const cfg = yaml.load(fs.readFileSync(configDir, 'utf8'))
  syncPort = cfg.syncPort
  dataDir = cfg.data
} catch (e) {
  throw(e)
}

const port = process.env.PORT || syncPort
if (port === undefined) {
  throw(Error('config file is missing required field nodePort'))
}

const app = express()
const httpServer = createServer(app)
const io = socketio(httpServer)

// maintain a store for each task
const stores: { [key: string]: Store } = {}

io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
  socket.on(EventName.REGISTER, (state: State) => {
    const taskId = state.task.config.taskId
    const projectName = state.task.config.projectName
    const workerId = state.user.id
    const sessionId = state.session.id

    const syncPath = path.getPath(dataDir, projectName, taskId, workerId)
    const room = path.roomName(projectName, taskId)
    // If there's no store in memory, try to load it
    if (!stores[room]) {
      /* frontend data is used to initialize a valid store
        but the non-task data is arbitrary */
      try {
        // Try to load task state from file system
        const fileNames = fs.readdirSync(syncPath)
        // Get most recently saved file
        const fileName = fileNames[fileNames.length - 1]
        const file = sprintf('%s/%s', syncPath, fileName)
        const rawContent = fs.readFileSync(file, 'utf8')
        const content = JSON.parse(rawContent)
        // Combine loaded task state with other state sent from frontend
        state.task = content
      } catch (e) {
        /* Don't crash if file does not exist; this just means it has not
          been initialized yet */
        if (e.code !== NO_FILE_ERROR_CODE) {
          throw(e)
        }
      }
      stores[room] = configureStore(state)
      // automatically save task data on updates
      stores[room].subscribe(() => {
        const content = JSON.stringify(stores[room].getState().present.task)
        const file = path.getFile(syncPath, sessionId)
        fs.writeFileSync(file, content)
      })
    } else {
      state.task = stores[room].getState().present.task
    }
    socket.join(room)
    // Send backend state to newly registered socket
    socket.emit(EventName.REGISTER_ACK, state)
  })

  socket.on(EventName.ACTION_SEND, (rawData: string) => {
    const data = JSON.parse(rawData)
    const taskId = data.id
    const projectName = data.project
    const workerId = data.worker
    const actionList = data.actions
    const syncPath = path.getPath(dataDir, projectName, taskId, workerId)
    // Make sure file exists before saving to it
    fs.ensureDirSync(syncPath)
    const room = path.roomName(projectName, taskId)
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

httpServer.listen(port)
