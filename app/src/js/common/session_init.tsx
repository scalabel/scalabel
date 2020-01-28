import { MuiThemeProvider } from '@material-ui/core/styles'
import _ from 'lodash'
import React from 'react'
import ReactDOM from 'react-dom'
import { Middleware } from 'redux'
import { sprintf } from 'sprintf-js'
import * as THREE from 'three'
import { addViewerConfig, initSessionAction, loadItem, updateAll } from '../action/common'
import Window from '../components/window'
import { makeDefaultViewerConfig } from '../functional/states'
import { State } from '../functional/types'
import { myTheme } from '../styles/theme'
import { PLYLoader } from '../thirdparty/PLYLoader'
import { configureStore } from './configure_store'
import Session from './session'
import { Synchronizer } from './synchronizer'
import { makeTrackPolicy, Track } from './track'
import { DataType, ViewerConfigTypeName } from './types'

/**
 * Request Session state from the server
 * @param {string} containerName - the name of the container
 */
export function initSession (containerName: string): void {
  // get params from url path. These uniquely identify a SAT.
  const searchParams = new URLSearchParams(window.location.search)
  const taskIndex = parseInt(searchParams.get('task_index') as string, 10)
  const projectName = searchParams.get('project_name') as string
  setListeners()

  const synchronizer = new Synchronizer(
    taskIndex,
    projectName,
    (state: State) => {
      initFromJson(state, synchronizer.middleware)
      renderDom(containerName, synchronizer)
    }
  )
}

/**
 * Update session objects with new state
 */
function updateTracks (): void {
  const state = Session.getState()
  const currentPolicyType =
    state.task.config.policyTypes[state.user.select.policyType]
  const newTracks: {[trackId: number]: Track} = {}
  for (const key of Object.keys(state.task.tracks)) {
    const trackId = Number(key)
    const track = state.task.tracks[trackId]
    if (trackId in Session.tracks) {
      newTracks[trackId] = Session.tracks[trackId]
      let trackPolicy = newTracks[trackId].trackPolicy
      if (newTracks[trackId].policyType !== currentPolicyType) {
        const newPolicy = makeTrackPolicy(newTracks[trackId], currentPolicyType)
        trackPolicy = newPolicy
      }
      newTracks[trackId].updateState(track, trackPolicy)
    } else {
      newTracks[trackId] = new Track()
      newTracks[trackId].updateState(
        track,
        makeTrackPolicy(newTracks[trackId], currentPolicyType)
      )
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
      <Window synchronizer={synchronizer} />
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
}

/**
 * Create default viewer configs if none exist
 */
function initViewerConfigs (): void {
  const state = Session.getState()
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
function setListeners () {
  const body = document.getElementsByTagName('BODY') as
    HTMLCollectionOf<HTMLElement>
  body[0].onresize = () => {
    Session.dispatch(updateAll())
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
function loadImages (): void {
  const state = Session.getState()
  const items = state.task.items
  Session.images = []
  for (const item of items) {
    const itemImageMap: {[id: number]: HTMLImageElement} = {}
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (sensorId in state.task.sensors &&
          state.task.sensors[sensorId].type === DataType.IMAGE) {
        const url = item.urls[sensorId]
        const image = new Image()
        image.crossOrigin = 'Anonymous'
        image.onload = () => {
          Session.dispatch(loadItem(item.index, sensorId))
        }
        image.onerror = () => {
          alert(sprintf('Failed to load image at %s', url))
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
function loadPointClouds (): void {
  const loader = new PLYLoader()

  const state = Session.getState()
  const items = state.task.items
  Session.pointClouds = []
  for (const item of items) {
    const pcImageMap: {[id: number]: THREE.BufferGeometry} = {}
    Session.pointClouds.push(pcImageMap)
    for (const key of Object.keys(item.urls)) {
      const sensorId = Number(key)
      if (sensorId in state.task.sensors &&
          state.task.sensors[sensorId].type === DataType.POINT_CLOUD) {
        const url = item.urls[sensorId]
        loader.load(
          url,
          (geometry: THREE.BufferGeometry) => {
            Session.pointClouds[item.index][sensorId] = geometry

            Session.dispatch(loadItem(item.index, sensorId))
          },

          () => null,

          () => {
            alert(`Point cloud at ${url} was not found.`)
          }
        )
      }
    }
  }
}
