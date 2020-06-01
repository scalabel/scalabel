import { MuiThemeProvider } from '@material-ui/core/styles'
import Fingerprint2 from 'fingerprintjs2'
import _ from 'lodash'
import React from 'react'
import ReactDOM from 'react-dom'
import { Provider } from 'react-redux'
import { Middleware } from 'redux'
import { sprintf } from 'sprintf-js'
import * as THREE from 'three'
import { addViewerConfig, connect, disconnect, initSessionAction,
  loadItem, receiveBroadcast, registerSession, setStatusAfterConnect,
  splitPane, updateAll, updatePane } from '../action/common'
import { alignToAxis, toggleSelectionLock } from '../action/point_cloud'
import Window from '../components/window'
import { makeDefaultViewerConfig } from '../functional/states'
import { PointCloudViewerConfigType, SplitType, State } from '../functional/types'
import { EventName, RegisterMessageType, SyncActionMessageType } from '../server/types'
import { myTheme } from '../styles/theme'
import { PLYLoader } from '../thirdparty/PLYLoader'
import { configureStore, ReduxStore } from './configure_store'
import Session from './session'
import { makeSyncMiddleware } from './sync_middleware'
import { Synchronizer } from './synchronizer'
import { Track } from './track/track'
import { DataType, ItemTypeName, QueryArg, ViewerConfigTypeName } from './types'

/**
 * Request Session state from the server
 * @param {string} containerName - the name of the container
 */
export function initSession (containerName: string): void {
  // Get params from url path. These uniquely identify a labeling task
  const searchParams = new URLSearchParams(window.location.search)
  const taskIndex = parseInt(
    searchParams.get(QueryArg.TASK_INDEX) as string, 10)
  const projectName = searchParams.get(QueryArg.PROJECT_NAME) as string

  // Wait for a second before getting the user fingerprint
  setTimeout(() => {
    Fingerprint2.get((components) => {
      const values =
        components.map((component) => component.value)
      const userId = Fingerprint2.x64hash128(values.join(''), 31)

      // Create middleware for handling sync actions
      const socket = io.connect(
        location.origin,
        { transports: ['websocket'], upgrade: false }
      )
      const synchronizer = new Synchronizer(
        socket, taskIndex, projectName, userId)
      const syncMiddleware = makeSyncMiddleware(synchronizer)

      // Override Session store to include the sync middleware
      const store = configureStore({}, Session.devMode, syncMiddleware)
      Session.store = store

      // Start the listeners that convert socket events to sync actions
      startSocketListeners(store, socket)
      setListeners(store)

      socket.on(EventName.CONNECT, () => {
        const message: RegisterMessageType = {
          projectName,
          taskIndex,
          sessionId: '',
          userId,
          address: location.origin,
          bot: false
        }
        /* Send the registration message to the backend */
        socket.emit(EventName.REGISTER, message)
        Session.dispatch(setStatusAfterConnect())
      })

      let firstTime = true
      socket.on(EventName.REGISTER_ACK, (state: State) => {
        if (firstTime) {
          firstTime = false

          // ideally has a refrence to the store
          // issue is middleware

          // middleware before store
          // store before sync
          // sendActions + queue before middleware
          // --> must remove those from sync
          // - can access store in middleware

          // TODO: create empty store initially to support consistent event handling

          // 1. Define handler which dispatches action for each socket event
          // 2, Create synchronizer for handling each event
          // 3. Middleware(sync) routes handler actions to sync methods
          const synchronizer = new Synchronizer(
            socket,
            taskIndex,
            projectName,
            userId,
            state.task.config.autosave,
            state.task.config.bots
          )
          // should overwrite the state via an action (keep the same store)
          initFromJson(state, synchronizer.middleware)
          renderDom(containerName, synchronizer)
        }
      })
    })
  }, 500)
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
 * Update session objects with new state
 */
export function updateTracks (): void {
  const state = Session.getState()
  const newTracks: {[trackId: string]: Track} = {}
  for (const trackId of Object.keys(state.task.tracks)) {
    if (trackId in Session.tracks) {
      newTracks[trackId] = Session.tracks[trackId]
    } else {
      const newTrack = new Track()
      if (newTrack) {
        newTracks[trackId] = newTrack
      }
    }
    if (trackId in newTracks) {
      newTracks[trackId].updateState(state, trackId)
    }
  }

  Session.tracks = newTracks
}

/**
 * Render the dom after data is laoded
 * @param containername: string name
 */
function renderDom (containerName: string, synchronizer: Synchronizer) {
  ReactDOM.render(
    <MuiThemeProvider theme={myTheme}>
      <Provider store={Session.store}>
        <Window
          synchronizer={synchronizer}
        />
      </Provider>
    </MuiThemeProvider>,
    document.getElementById(containerName))
}

/**
 * Initialize state store
 * @param {{}}} stateJson: json state from backend
 * @param middleware: optional middleware for redux
 */
export function initStore (stateJson: {}, middleware?: Middleware): void {
  Session.store = configureStore(stateJson, Session.devMode, middleware)
  Session.dispatch(initSessionAction())
  const state = Session.getState()
  Session.tracking = state.task.config.tracking
  Session.autosave = state.task.config.autosave
  Session.bots = state.task.config.bots
}

/**
 * Create default viewer configs if none exist
 */
function initViewerConfigs (): void {
  let state = Session.getState()
  if (Object.keys(state.user.viewerConfigs).length === 0) {
    const sensorIds = Object.keys(state.task.sensors).map(
      (key) => Number(key)
    ).sort()
    const id0 = sensorIds[0]
    const sensor0 = state.task.sensors[id0]
    const config0 = makeDefaultViewerConfig(
      sensor0.type as ViewerConfigTypeName, 0, id0
    )
    if (config0) {
      Session.dispatch(addViewerConfig(0, config0))
    }

    // Set up default PC labeling interface
    const paneIds = Object.keys(state.user.layout.panes)
    if (
      state.task.config.itemType === ItemTypeName.POINT_CLOUD &&
      paneIds.length === 1
    ) {
      Session.dispatch(splitPane(Number(paneIds[0]), SplitType.HORIZONTAL, 0))
      state = Session.getState()
      let config =
        state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      Session.dispatch(toggleSelectionLock(
        state.user.layout.maxViewerConfigId,
        config as PointCloudViewerConfigType
      ))
      Session.dispatch(splitPane(
        state.user.layout.maxPaneId,
        SplitType.VERTICAL,
        state.user.layout.maxViewerConfigId
      ))
      Session.dispatch(updatePane(
        state.user.layout.maxPaneId, { primarySize: '33%' }
      ))

      state = Session.getState()
      config =
        state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      Session.dispatch(toggleSelectionLock(
        state.user.layout.maxViewerConfigId,
        config as PointCloudViewerConfigType
      ))
      Session.dispatch(alignToAxis(
        state.user.layout.maxViewerConfigId,
        config as PointCloudViewerConfigType,
        1
      ))
      Session.dispatch(splitPane(
        state.user.layout.maxPaneId,
        SplitType.VERTICAL,
        state.user.layout.maxViewerConfigId
      ))

      state = Session.getState()
      config =
        state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      Session.dispatch(toggleSelectionLock(
        state.user.layout.maxViewerConfigId,
        config as PointCloudViewerConfigType
      ))
      Session.dispatch(alignToAxis(
        state.user.layout.maxViewerConfigId,
        config as PointCloudViewerConfigType,
        2
      ))
    }
  }
}

/**
 * Init general labeling session.
 * @param {{}}} stateJson: json state from backend
 * @param middleware: optional middleware for redux
 */
export function initFromJson (stateJson: {}, middleware?: Middleware): void {
  initStore(stateJson, middleware)
  initViewerConfigs()
  loadData()
  Session.subscribe(updateTracks)
  Session.dispatch(updateAll())
  Session.subscribe(() => Session.label3dList.updateState(Session.getState()))
  Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
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

/**
 * Load labeling data initialization function
 */
function loadData (): void {
  loadImages()
  loadPointClouds()
}

/**
 * Load all the images in the state
 */
export function loadImages (maxAttempts: number = 3): void {
  const state = Session.getState()
  const items = state.task.items
  Session.images = []
  for (const item of items) {
    const itemImageMap: {[id: number]: HTMLImageElement} = {}
    const attemptsMap: {[id: number]: number} = {}
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (sensorId in state.task.sensors &&
          state.task.sensors[sensorId].type === DataType.IMAGE) {
        attemptsMap[sensorId] = 0
        const url = item.urls[sensorId]
        const image = new Image()
        image.crossOrigin = 'Anonymous'
        image.onload = () => {
          Session.dispatch(loadItem(item.index, sensorId))
        }
        image.onerror = () => {
          if (attemptsMap[sensorId] === maxAttempts) {
            // Append date to url to prevent local caching
            image.src = `${url}#${new Date().getTime()}`
            attemptsMap[sensorId]++
          } else {
            alert(sprintf('Failed to load image at %s', url))
          }
        }
        image.src = url
        itemImageMap[sensorId] = image
      }
    }
    Session.images.push(itemImageMap)
  }
}

/**
 * Load all point clouds in state
 */
function loadPointClouds (maxAttempts: number = 3): void {
  const loader = new PLYLoader()

  const state = Session.getState()
  const items = state.task.items
  Session.pointClouds = []
  for (const item of items) {
    const pcImageMap: {[id: number]: THREE.BufferGeometry} = {}
    Session.pointClouds.push(pcImageMap)
    const attemptsMap: {[id: number]: number} = {}
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (sensorId in state.task.sensors &&
          state.task.sensors[sensorId].type === DataType.POINT_CLOUD) {
        const url = item.urls[sensorId]
        attemptsMap[sensorId] = 0
        const onLoad = (geometry: THREE.BufferGeometry) => {
          Session.pointClouds[item.index][sensorId] = geometry

          Session.dispatch(loadItem(item.index, sensorId))
        }
        // TODO(fyu): need to make a unified data loader with consistent
        // policy for all data types
        const onError = () => {
          attemptsMap[sensorId]++
          if (attemptsMap[sensorId] === maxAttempts) {
            alert(`Point cloud at ${url} was not found.`)
          } else {
            loader.load(
              url,
              onLoad,
              () => null,
              onError
            )
          }
        }
        loader.load(
          url,
          onLoad,
          () => null,
          onError
        )
      }
    }
  }
}
