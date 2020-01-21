import * as socketio from 'socket.io'
import * as uuid4 from 'uuid/v4'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { makeItemStatus, makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import * as path from './path'
import { RedisCache } from './redis_cache'
import Session from './server_session'
import { EventName } from './types'
import { getSavedKey, getTaskKey,
  index2str, loadSavedState} from './util'

/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (io: socketio.Server) {
  const cache = new RedisCache()

  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      await register(rawData, socket, cache)
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      await actionUpdate(rawData, socket, cache, io)
    })
  })
}

/**
 * Load the correct state and subscribe to redis
 */
async function register (
  rawData: string, socket: socketio.Socket, cache: RedisCache) {
  const data = JSON.parse(rawData)
  const projectName = data.project
  const taskIndex = data.index
  let sessId = data.sessId
  const env = Session.getEnv()

  const taskId = index2str(taskIndex)
  // keep session id if it exists, i.e. if it is a reconnection
  if (!sessId) {
    // new session on new load
    sessId = uuid4()
  }
  const room = path.roomName(projectName, taskId, env.sync, sessId)

  const state = await loadState(room, projectName, taskId, cache)
  state.session.id = sessId
  state.task.config.autosave = env.autosave

  // Connect socket to others in the same room
  socket.join(room)
  // Send backend state to newly registered socket
  socket.emit(EventName.REGISTER_ACK, state)
}

/**
 * Updates the state with the action, and broadcasts action
 */
async function actionUpdate (
  rawData: string, socket: socketio.Socket,
  cache: RedisCache, io: socketio.Server) {
  const data = JSON.parse(rawData)
  const projectName = data.project
  const taskId = data.taskId
  const sessId = data.sessId
  const actionList = data.actions
  const env = Session.getEnv()

  const room = path.roomName(projectName, taskId, env.sync, sessId)
  const state = await loadState(room, projectName, taskId, cache)
  const store = configureStore(state)

  // For each action, update the backend store and broadcast
  for (const action of actionList) {
    action.timestamp = Date.now()
    // for task actions, update store and broadcast to room
    if (types.TASK_ACTION_TYPES.includes(action.type)) {
      store.dispatch(action)
      io.in(room).emit(EventName.ACTION_BROADCAST, action)
    } else {
      socket.emit(EventName.ACTION_BROADCAST, action)
    }
  }

  const newState = store.getState().present
  const stringState = JSON.stringify(newState)
  const filePath = path.getFileKey(getSavedKey(projectName, taskId))

  await cache.setExWithReminder(room, filePath, stringState)
}

/**
 * Loads state from cache if available, else memory
 */
async function loadState (
  room: string, projectName: string, taskId: string,
  cache: RedisCache): Promise<State> {
  let state: State

  // first try to load from redis cache
  const redisValue = await cache.get(room)
  if (redisValue) {
    state = JSON.parse(redisValue)
  } else {
    // otherwise load from storage
    try {
      // first, attempt loading previous submission
      state = await loadSavedState(projectName, taskId)
    } catch {
      // if no submissions exist, load from task
      state = await loadStateFromTask(projectName, taskId)
    }
  }
  return state
}

/**
 * Loads the state from task.json (created at import)
 * Used for first load
 */
async function loadStateFromTask (
  projectName: string,
  taskId: string): Promise<State> {
  const key = getTaskKey(projectName, taskId)
  const fields = await Session.getStorage().load(key)
  const task = JSON.parse(fields) as TaskType
  const state = makeState({ task })

  state.session.itemStatuses = []
  for (const item of state.task.items) {
    const itemStatus = makeItemStatus()
    for (const sensorKey of Object.keys(item.urls)) {
      itemStatus.sensorDataLoaded[Number(sensorKey)] = false
    }
    state.session.itemStatuses.push(itemStatus)
  }

  switch (state.task.config.itemType) {
    case ItemTypeName.IMAGE:
    case ItemTypeName.VIDEO:
      if (state.task.config.labelTypes.length === 1 &&
          state.task.config.labelTypes[0] === LabelTypeName.BOX_2D) {
        state.task.config.policyTypes =
          [TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D]
      }
      break
    case ItemTypeName.POINT_CLOUD:
    case ItemTypeName.POINT_CLOUD_TRACKING:
      if (state.task.config.labelTypes.length === 1 &&
          state.task.config.labelTypes[0] === LabelTypeName.BOX_3D) {
        state.task.config.policyTypes =
          [TrackPolicyType.LINEAR_INTERPOLATION_BOX_3D]
      }
      break
  }
  return state
}