import { MuiThemeProvider } from '@material-ui/core/styles'
import _ from 'lodash'
import React from 'react'
import ReactDOM from 'react-dom'
import { Middleware } from 'redux'
import { sprintf } from 'sprintf-js'
import * as THREE from 'three'
import { initSessionAction, loadItem, updateAll } from '../action/common'
import Window from '../components/window'
import {
  makeImageViewerConfig,
  makePointCloudViewerConfig
} from '../functional/states'
import {
  ImageViewerConfigType,
  PointCloudViewerConfigType,
  State
} from '../functional/types'
import { myTheme } from '../styles/theme'
import { PLYLoader } from '../thirdparty/PLYLoader'
import { configureStore } from './configure_store'
import Session from './session'
import { Synchronizer } from './synchronizer'
import { makeTrackPolicy, Track } from './track'

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
  Session.currentPolicyType = currentPolicyType
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
  Session.itemType = state.task.config.itemType
  Session.tracking = state.task.config.tracking
  Session.autosave = state.task.config.autosave
}

/**
 * Init general labeling session.
 * @param {{}}} stateJson: json state from backend
 * @param middleware: optional middleware for redux
 */
export function initFromJson (stateJson: {}, middleware?: Middleware): void {
  initStore(stateJson, middleware)
  loadData()
  Session.subscribe(updateTracks)
  Session.dispatch(updateAll())
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
  if (Session.itemType === 'image' || Session.itemType === 'video') {
    loadImages()
  } else if (Session.itemType === 'pointcloud') {
    loadPointClouds()
  }
}

/**
 * Load all the images in the state
 */
function loadImages (): void {
  const state = Session.getState()
  const items = state.task.items
  for (const item of items) {
    // Copy item config
    let config: ImageViewerConfigType = {
      ...(state.user.imageViewerConfig)
    }
    if (_.isEmpty(config)) {
      config = makeImageViewerConfig()
    }
    const url = item.url
    const image = new Image()
    image.crossOrigin = 'Anonymous'
    Session.images.push(image)
    image.onload = () => {
      config.imageHeight = image.height
      config.imageWidth = image.width
      Session.dispatch(loadItem(item.index, config))
    }
    image.onerror = () => {
      alert(sprintf('Failed to load image at %s', url))
    }
    image.src = url
  }
}

/**
 * Load all point clouds in state
 */
function loadPointClouds (): void {
  const loader = new PLYLoader()
  const vertexShader =
    `
      varying float distFromOrigin;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
        distFromOrigin = length(position);
        gl_PointSize = 0.1 * ( 300.0 / -mvPosition.z );
        gl_Position = projectionMatrix * mvPosition;
      }
    `
  const fragmentShader =
    `
      varying float distFromOrigin;
      uniform vec3 red;
      uniform vec3 yellow;
      uniform vec3 green;
      uniform vec3 teal;
      vec3 getHeatMapColor(float dist) {
        if (dist < 8.0) {
          float val = dist / 8.0;
          return (1.0 - val) * red + val * yellow;
        } else if (dist < 16.0) {
          float val = (dist - 8.0) / 8.0;
          return (1.0 - val) * yellow + val * green;
        } else {
          float val = (dist - 16.0) / 8.0;
          return (1.0 - val) * green + val * teal;
        }
      }
      void main() {
        gl_FragColor = vec4(getHeatMapColor(distFromOrigin), 1.0);
      }
    `

  const state = Session.getState()
  const items = state.task.items
  for (const item of items) {
    let config: PointCloudViewerConfigType = {
      ...(state.user.pointCloudViewerConfig)
    }
    if (_.isEmpty(config)) {
      config = makePointCloudViewerConfig()
    }
    Session.pointClouds.push(new THREE.Points())
    loader.load(item.url, (geometry: THREE.BufferGeometry) => {

      const material = new THREE.ShaderMaterial({
        uniforms: {
          red: {
            value: new THREE.Color(0xff0000)
          },
          yellow: {
            value: new THREE.Color(0xffff00)
          },
          green: {
            value: new THREE.Color(0x00ff00)
          },
          teal: {
            value: new THREE.Color(0x00ffff)
          }
        },
        vertexShader,
        fragmentShader,
        alphaTest: 1.0
      })

      const particles = new THREE.Points(geometry, material)
      Session.pointClouds[item.index] = particles

      Session.dispatch(loadItem(item.index, config))
    },

      () => null,

      () => {
        alert('Point cloud at ' + item.url + ' was not found.')
      }
    )
  }
}
