import { Store } from 'redux'
import * as socketio from 'socket.io'
import * as uuid4 from 'uuid/v4'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import * as path from './path'
import Session from './server_session'
import { EventName } from './types'
import { getSavedKey, getTaskKey,
  index2str, loadSavedState} from './util'

/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (io: socketio.Server) {
  const env = Session.getEnv()
  // maintain a store for each task
  const stores: { [key: string]: Store } = {}
  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      const data = JSON.parse(rawData)
      const projectName = data.project

      const taskIndex = data.index
      const taskId = index2str(taskIndex)

      let sessId = data.sessId
      // keep session id if it exists, i.e. if it is a reconnection
      if (!sessId) {
        // new session on new load
        sessId = uuid4()
      }

      let room = path.roomName(projectName, taskId, env.sync)

      let state: State

      // if sync is on, try to load from memory
      if (env.sync && room in stores) {
        // try load from memory
        state = stores[room].getState().present
      } else {
        // load from storage
        try {
          // first, attempt loading previous submission
          state = await loadSavedState(projectName, taskId)
        } catch {
          // if no submissions exist, load from task
          state = await loadStateFromTask(projectName, taskIndex)
        }

        // update room with loaded sessId
        room = path.roomName(projectName, taskId,
          env.sync, sessId)

        // update memory with new state
        stores[room] = configureStore(state)

        // automatically save task data on updates
        stores[room].subscribe(async () => {
          const content = JSON.stringify(stores[room].getState().present)
          const filePath = path.getFileKey(getSavedKey(projectName, taskId))
          await Session.getStorage().save(filePath, content)
        })
      }
      state.session.id = sessId
      state.task.config.autosave = env.autosave

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

      const room = path.roomName(projectName, taskId, env.sync, sessId)

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

/**
 * Loads the state from task.json (created at import)
 * Used for first load
 */
async function loadStateFromTask (
  projectName: string,
  taskIndex: number): Promise<State> {
  const key = getTaskKey(projectName, index2str(taskIndex))
  const fields = await Session.getStorage().load(key)
  const task = JSON.parse(fields) as TaskType
  const state = makeState({ task })

  state.session.items = state.task.items.map((_i) => ({ loaded: false }))

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
