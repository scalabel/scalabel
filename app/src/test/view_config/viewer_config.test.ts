import _ from 'lodash'
// import * as THREE from 'three'
import * as action from '../../js/action/common'
import { MOUSE_CORRECTION_FACTOR, moveCameraAndTarget } from '../../js/action/point_cloud'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { getCurrentViewerConfig } from '../../js/functional/state_util'
import { makePointCloudViewerConfig } from '../../js/functional/states'
import { PointCloudViewerConfigType, Vector3Type } from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import ViewerConfigUpdater from '../../js/view_config/viewer_config'
import { testJson } from '../test_point_cloud_objects'

/**
 * Check equality between two Vector3Type objects
 * @param v1
 * @param v2
 */
function expectVector3TypesClose (v1: Vector3Type, v2: Vector3Type) {
  expect(v1.x).toBeCloseTo(v2.x)
  expect(v1.y).toBeCloseTo(v2.y)
  expect(v1.z).toBeCloseTo(v2.z)
}

test('Viewer Config 3d drag test', () => {
  Session.devMode = false
  initStore(testJson)
  Session.dispatch(action.addViewerConfig(1, makePointCloudViewerConfig(-1)))
  const viewerId = 1
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  let state = Session.getState()
  let viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  const viewerConfigUpdater = new ViewerConfigUpdater()
  Session.subscribe(() => {
    viewerConfigUpdater.updateState(Session.getState(), viewerId)
  })
  viewerConfigUpdater.setContainer(document.createElement('div'))

  Session.dispatch(moveCameraAndTarget(
    (new Vector3D()).fromObject({ x: 0, y: 1, z: 0 }),
    (new Vector3D()).fromObject({ x: 0, y: 0, z: 0 }),
    viewerId,
    viewerConfig
  ))

  state = Session.getState()
  viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  expectVector3TypesClose(viewerConfig.position, { x: 0, y: 1, z: 0 })
  expectVector3TypesClose(viewerConfig.target, { x: 0, y: 0, z: 0 })

  viewerConfigUpdater.onMouseDown(0, 0, 2)

  viewerConfigUpdater.onMouseMove(
    MOUSE_CORRECTION_FACTOR / 2,
    MOUSE_CORRECTION_FACTOR / 2
  )

  viewerConfigUpdater.onMouseUp()

  state = Session.getState()
  viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  expectVector3TypesClose(viewerConfig.position, { x: 1, y: 1, z: 1 })
  expectVector3TypesClose(viewerConfig.target, { x: 1, y: 0, z: 1 })
})
