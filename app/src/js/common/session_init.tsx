import Fingerprint2 from 'fingerprintjs2'
import _ from 'lodash'
import io from 'socket.io-client'
import { connect, disconnect, receiveBroadcast, registerSession,
  updateAll } from '../action/common'
import { State } from '../functional/types'
import { EventName, SyncActionMessageType } from '../server/types'
import { configureStore, ReduxStore } from './configure_store'
import Session from './session'
import { makeSyncMiddleware } from './sync_middleware'
import { Synchronizer } from './synchronizer'
import { QueryArg } from './types'

/**
 * Main function- initiates frontend session
 * @param {string} containerName - the name of the container
 */
export function initSession (containerName: string): void {
  // Get params from url path. These uniquely identify a labeling task
  const searchParams = new URLSearchParams(window.location.search)
  const taskIndex = parseInt(
    searchParams.get(QueryArg.TASK_INDEX) as string, 10)
  const projectName = searchParams.get(QueryArg.PROJECT_NAME) as string

  /**
   * Wait for page to load to ensure consistent fingerprint
   * See docs at https://github.com/Valve/fingerprintjs2
   */
  setTimeout(() => {
    Fingerprint2.get((components) => {
      const values =
        components.map((component) => component.value)
      const userId = Fingerprint2.x64hash128(values.join(''), 31)
      initGenericSession(taskIndex, projectName, userId, containerName)
    })
  }, 500)
}

/**
 * Inits session given the parameters
 */
export function initGenericSession (
  taskIndex: number, projectName: string,
  userId: string, containerName: string) {
  // Create middleware for handling sync actions
  const socket = io.connect(
    location.origin,
    { transports: ['websocket'], upgrade: false }
  )
  const synchronizer = new Synchronizer(
    socket, taskIndex, projectName, userId, containerName)
  const syncMiddleware = makeSyncMiddleware(synchronizer)

  // Initialize empty store
  const store = configureStore({}, Session.devMode, syncMiddleware)
  Session.store = store

  // Start the listeners that convert socket events to sync actions
  // This will handle loading the initial state data
  startSocketListeners(store, socket)
  setListeners(store)
}

/**
 * Connect socket events to Redux actions
 */
export function startSocketListeners (
  store: ReduxStore, socket: SocketIOClient.Socket) {
  socket.on(EventName.CONNECT, () => {
    store.dispatch(connect())
  })
  socket.on(EventName.DISCONNECT, () => {
    store.dispatch(disconnect())
  })
  socket.on(EventName.REGISTER_ACK, (state: State) => {
    store.dispatch(registerSession(state))
  })
  socket.on(EventName.ACTION_BROADCAST, (message: SyncActionMessageType) => {
    store.dispatch(receiveBroadcast(message))
  })
}

/**
 * Set listeners for the html body
 */
function setListeners (store: ReduxStore) {
  const body = document.getElementsByTagName('BODY') as
    HTMLCollectionOf<HTMLElement>
  body[0].onresize = () => {
    store.dispatch(updateAll())
  }
}
