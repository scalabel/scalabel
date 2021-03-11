import Fingerprint2 from "fingerprintjs2"
import io from "socket.io-client"

import {
  connect,
  disconnect,
  receiveBroadcast,
  registerSession,
  updateAll
} from "../action/common"
import { QueryArg } from "../const/common"
import { EventName } from "../const/connection"
import { SyncActionMessageType } from "../types/message"
import { FullStore } from "../types/redux"
import { State } from "../types/state"
import { configureStore } from "./configure_store"
import Session from "./session"
import { makeSyncMiddleware } from "./sync_middleware"
import { Synchronizer } from "./synchronizer"
import { handleInvalidPage } from "./util"

/**
 * Main function for initiating the frontend session
 *
 * @param {string} containerName - the name of the container
 */
export function initSession(containerName: string): void {
  // Get params from url path. These uniquely identify a labeling task
  const searchParams = new URLSearchParams(window.location.search)
  const projectName = searchParams.get(QueryArg.PROJECT_NAME)
  if (projectName === null) {
    return handleInvalidPage()
  }
  const taskIndexParam = searchParams.get(QueryArg.TASK_INDEX)
  let taskIndex = 0
  if (taskIndexParam !== null) {
    taskIndex = parseInt(taskIndexParam, 10)
  }
  const devMode = searchParams.has(QueryArg.DEV_MODE)

  /**
   * Wait for page to load to ensure consistent fingerprint
   * See docs at https://github.com/Valve/fingerprintjs2
   */
  setTimeout(() => {
    Fingerprint2.get((components) => {
      const values = components.map((component) => component.value)
      const userId = Fingerprint2.x64hash128(values.join(""), 31)
      initSessionForTask(taskIndex, projectName, userId, containerName, devMode)
    })
  }, 500)
}

/**
 * Initializes a frontend session given the task's identifying information
 *
 * @param taskIndex
 * @param projectName
 * @param userId
 * @param containerName
 * @param devMode
 */
export function initSessionForTask(
  taskIndex: number,
  projectName: string,
  userId: string,
  containerName: string,
  devMode: boolean
): void {
  // Initialize socket connection to the backend
  const socket = io.connect(location.origin, {
    transports: ["websocket"],
    upgrade: false
  })

  // Create middleware for handling socket.io-based actions
  const synchronizer = new Synchronizer(
    socket,
    taskIndex,
    projectName,
    userId,
    containerName
  )
  const syncMiddleware = makeSyncMiddleware(synchronizer)

  // Initialize empty store
  const store = configureStore({}, devMode, syncMiddleware)
  Session.store = store

  // Start the listeners that convert socket.io events to Redux actions
  // These listeners will handle loading of the initial state data
  setSocketListeners(store, socket)

  // Set HTML listeners
  setBodyListeners(store)
}

/**
 * Connect socket events to Redux actions
 *
 * @param store
 * @param socket
 */
export function setSocketListeners(
  store: FullStore,
  socket: SocketIOClient.Socket
): void {
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
 * Set listeners for the HTML body
 *
 * @param store
 */
function setBodyListeners(store: FullStore): void {
  const body = document.getElementsByTagName("BODY") as HTMLCollectionOf<
    HTMLElement
  >
  body[0].onresize = () => {
    store.dispatch(updateAll())
  }
}
