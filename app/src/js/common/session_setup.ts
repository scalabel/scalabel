import { sprintf } from 'sprintf-js'
import * as THREE from 'three'
import { addViewerConfig, initSessionAction, loadItem,
  splitPane, updateAll, updatePane } from '../action/common'
import { alignToAxis, toggleSelectionLock } from '../action/point_cloud'
import { makeDefaultViewerConfig } from '../functional/states'
import { PointCloudViewerConfigType, SplitType } from '../functional/types'
import { PLYLoader } from '../thirdparty/PLYLoader'
import Session from './session'
import { Track } from './track/track'
import { DataType, ItemTypeName, ViewerConfigTypeName } from './types'

/**
 * After the state is initialized, set up the rest of the session
 */
export function setupSession (shouldInitViews: boolean) {
  Session.dispatch(initSessionAction())
  const state = Session.getState()
  Session.tracking = state.task.config.tracking
  Session.autosave = state.task.config.autosave
  Session.bots = state.task.config.bots

  if (shouldInitViews) {
    initViewerConfigs()
    loadData()
    Session.subscribe(updateTracks)
    Session.dispatch(updateAll())
    Session.subscribe(() => Session.label3dList.updateState(Session.getState()))
    Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
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
