import * as socketio from 'socket.io'
import * as uuid4 from 'uuid/v4'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import Logger from './logger'
import * as path from './path'
import { ProjectStore } from './project_store'
import {
  Env, EventName, RegisterMessageType, SyncActionMessageType } from './types'
import { index2str } from './util'

/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (
  io: socketio.Server, env: Env, projectStore: ProjectStore) {
  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      try {
        await register(rawData, socket, env, projectStore)
      } catch (error) {
        Logger.error(error)
      }
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      try {
        await actionUpdate(rawData, socket, env, projectStore)
      } catch (error) {
        Logger.error(error)
      }
    })
  })
}

/**
 * Load the correct state and subscribe to redis
 */
async function register (
  rawData: string, socket: socketio.Socket,
  env: Env, projectStore: ProjectStore) {
  const data: RegisterMessageType = JSON.parse(rawData)
  const projectName = data.projectName
  const taskIndex = data.taskIndex
  let sessionId = data.sessionId

  const taskId = index2str(taskIndex)
  // keep session id if it exists, i.e. if it is a reconnection
  if (!sessionId) {
    // new session on new load
    sessionId = uuid4()
  }
  const state = await projectStore.loadState(projectName, taskId)
  state.session.id = sessionId
  state.task.config.autosave = env.autosave

  // Connect socket to others in the same room
  const room = path.getRoomName(projectName, taskId, env.sync, sessionId)
  socket.join(room)
  // Send backend state to newly registered socket
  socket.emit(EventName.REGISTER_ACK, state)
}

/**
 * Updates the state with the action, and broadcasts action
 */
async function actionUpdate (
  rawData: string, socket: socketio.Socket,
  env: Env, projectStore: ProjectStore) {
  const data: SyncActionMessageType = JSON.parse(rawData)
  const projectName = data.projectName
  const taskId = data.taskId
  const sessionId = data.sessionId
  const actionList = data.actions

  const room = path.getRoomName(projectName, taskId, env.sync, sessionId)

  const taskActions = actionList.filter((action) => {
    return types.TASK_ACTION_TYPES.includes(action.type)
  })

  const state = await projectStore.loadState(projectName, taskId)
  const stateStore = configureStore(state)

  // For each task action, update the backend store
  for (const action of taskActions) {
    action.timestamp = Date.now()
    stateStore.dispatch(action)
  }

  // broadcast task actions to all other sessions in room
  socket.broadcast.to(room).emit(EventName.ACTION_BROADCAST, taskActions)
  // echo everything to original session
  socket.emit(EventName.ACTION_BROADCAST, actionList)

  const newState = stateStore.getState().present

  await projectStore.saveState(newState, projectName, taskId)
}
