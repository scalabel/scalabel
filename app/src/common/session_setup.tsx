import { CssBaseline } from "@material-ui/core"
import { ThemeProvider } from "@material-ui/core/styles"
import React from "react"
import ReactDOM from "react-dom"
import { Provider } from "react-redux"
import * as THREE from "three"

import {
  addViewerConfig,
  initSessionAction,
  loadItem,
  splitPane,
  updateAll,
  updatePane,
  updateState
} from "../action/common"
import { alignToAxis, toggleSelectionLock } from "../action/point_cloud"
import Window from "../components/window"
import { DataType, ItemTypeName, ViewerConfigTypeName } from "../const/common"
import { makeDefaultViewerConfig } from "../functional/states"
import { scalabelTheme } from "../styles/theme"
import { PLYLoader } from "../thirdparty/PLYLoader"
import { FullStore } from "../types/redux"
import {
  DeepPartialState,
  PointCloudViewerConfigType,
  SplitType,
  State
} from "../types/state"
import Session from "./session"
import { DispatchFunc, GetStateFunc } from "./simple_store"
import { Track } from "./track"

/**
 * Initialize state, then set up the rest of the session
 *
 * @param newState
 * @param containerName
 * @param shouldInitViews
 */
export function setupSession(
  newState: DeepPartialState,
  containerName: string = "",
  shouldInitViews: boolean = true
): void {
  const store = Session.getSimpleStore()
  const dispatch = store.dispatcher()
  const getState = store.getter()

  // Update with the state from the backend
  dispatch(updateState(newState))
  dispatch(initSessionAction())

  // Unless in testing mode, update views
  if (shouldInitViews && containerName !== "") {
    initViewerConfigs(getState, dispatch)
    loadData(getState, dispatch)
    Session.subscribe(() => updateTracks(getState()))
    dispatch(updateAll())
    Session.subscribe(() => Session.label3dList.updateState(getState()))
    Session.subscribe(() => Session.label2dList.updateState(getState()))
    renderDom(containerName, Session.store)
  }
}

/**
 * Render the dom after data is loaded
 *
 * @param containername: string name
 * @param containerName
 * @param store
 */
function renderDom(containerName: string, store: FullStore): void {
  ReactDOM.render(
    <ThemeProvider theme={scalabelTheme}>
      <CssBaseline />
      <Provider store={store}>
        <Window />
      </Provider>
    </ThemeProvider>,
    document.getElementById(containerName)
  )
}

/**
 * Load labeling data initialization function
 *
 * @param getState
 * @param dispatch
 */
function loadData(getState: GetStateFunc, dispatch: DispatchFunc): void {
  loadImages(getState, dispatch)
  loadPointClouds(getState, dispatch)
}

/**
 * Update session objects with new state
 *
 * @param state
 */
export function updateTracks(state: State): void {
  const newTracks: { [trackId: string]: Track } = {}
  for (const trackId of Object.keys(state.task.tracks)) {
    if (trackId in Session.tracks) {
      newTracks[trackId] = Session.tracks[trackId]
    } else {
      const newTrack = new Track()
      newTracks[trackId] = newTrack
    }
    if (trackId in newTracks) {
      newTracks[trackId].updateState(state, trackId)
    }
  }

  Session.tracks = newTracks
}

/**
 * Load all the images in the state
 *
 * @param getState
 * @param dispatch
 * @param maxAttempts
 */
function loadImages(
  getState: GetStateFunc,
  dispatch: DispatchFunc,
  maxAttempts: number = 3
): void {
  const state = getState()
  const items = state.task.items
  Session.images = []
  for (const item of items) {
    const itemImageMap: { [id: number]: HTMLImageElement } = {}
    const attemptsMap: { [id: number]: number } = {}
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (
        sensorId in state.task.sensors &&
        state.task.sensors[sensorId].type === DataType.IMAGE
      ) {
        attemptsMap[sensorId] = 0
        const url = item.urls[sensorId]
        const image = new Image()
        image.crossOrigin = "Anonymous"
        image.onload = () => {
          dispatch(loadItem(item.index, sensorId))
        }
        image.onerror = () => {
          if (attemptsMap[sensorId] === maxAttempts) {
            // Append date to url to prevent local caching
            image.src = `${url}#${new Date().getTime()}`
            attemptsMap[sensorId]++
          } else {
            alert(`Failed to load image at ${url}`)
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
 *
 * @param getState
 * @param dispatch
 * @param maxAttempts
 */
function loadPointClouds(
  getState: GetStateFunc,
  dispatch: DispatchFunc,
  maxAttempts: number = 3
): void {
  const loader = new PLYLoader()

  const state = getState()
  const items = state.task.items
  Session.pointClouds = []
  for (const item of items) {
    const pcImageMap: { [id: number]: THREE.BufferGeometry } = {}
    Session.pointClouds.push(pcImageMap)
    const attemptsMap: { [id: number]: number } = {}
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (
        sensorId in state.task.sensors &&
        state.task.sensors[sensorId].type === DataType.POINT_CLOUD
      ) {
        const url = item.urls[sensorId]
        attemptsMap[sensorId] = 0
        const onLoad = (geometry: THREE.BufferGeometry): void => {
          Session.pointClouds[item.index][sensorId] = geometry

          dispatch(loadItem(item.index, sensorId))
        }
        // TODO(fyu): need to make a unified data loader with consistent
        // policy for all data types
        const onError = (): void => {
          attemptsMap[sensorId]++
          if (attemptsMap[sensorId] === maxAttempts) {
            alert(`Point cloud at ${url} was not found.`)
          } else {
            loader.load(url, onLoad, () => null, onError)
          }
        }
        loader.load(url, onLoad, () => null, onError)
      }
    }
  }
}

/**
 * Create default viewer configs if none exist
 *
 * @param getState
 * @param dispatch
 */
function initViewerConfigs(
  getState: GetStateFunc,
  dispatch: DispatchFunc
): void {
  let state = getState()
  if (Object.keys(state.user.viewerConfigs).length === 0) {
    const sensorIds = Object.keys(state.task.sensors)
      .map((key) => Number(key))
      .sort((a, b) => a - b)
    const id0 = sensorIds[0]
    const sensor0 = state.task.sensors[id0]
    const config0 = makeDefaultViewerConfig(
      sensor0.type as ViewerConfigTypeName,
      0,
      id0
    )
    if (config0 !== null) {
      dispatch(addViewerConfig(0, config0))
    }

    // Set up default PC labeling interface
    const paneIds = Object.keys(state.user.layout.panes)
    if (
      state.task.config.itemType === ItemTypeName.POINT_CLOUD &&
      paneIds.length === 1
    ) {
      dispatch(splitPane(Number(paneIds[0]), SplitType.HORIZONTAL, 0))
      state = getState()
      let config = state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      dispatch(
        toggleSelectionLock(
          state.user.layout.maxViewerConfigId,
          config as PointCloudViewerConfigType
        )
      )
      dispatch(
        splitPane(
          state.user.layout.maxPaneId,
          SplitType.VERTICAL,
          state.user.layout.maxViewerConfigId
        )
      )
      dispatch(updatePane(state.user.layout.maxPaneId, { primarySize: "33%" }))

      state = getState()
      config = state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      dispatch(
        toggleSelectionLock(
          state.user.layout.maxViewerConfigId,
          config as PointCloudViewerConfigType
        )
      )
      dispatch(
        alignToAxis(
          state.user.layout.maxViewerConfigId,
          config as PointCloudViewerConfigType,
          1
        )
      )
      dispatch(
        splitPane(
          state.user.layout.maxPaneId,
          SplitType.VERTICAL,
          state.user.layout.maxViewerConfigId
        )
      )

      state = getState()
      config = state.user.viewerConfigs[state.user.layout.maxViewerConfigId]
      dispatch(
        toggleSelectionLock(
          state.user.layout.maxViewerConfigId,
          config as PointCloudViewerConfigType
        )
      )
      dispatch(
        alignToAxis(
          state.user.layout.maxViewerConfigId,
          config as PointCloudViewerConfigType,
          2
        )
      )
    }
  }
}
