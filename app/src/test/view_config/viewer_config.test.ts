import _ from 'lodash'
// import * as THREE from 'three'
import * as action from '../../js/action/common'
import { MOUSE_CORRECTION_FACTOR, moveCameraAndTarget } from '../../js/action/point_cloud'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { getCurrentPointCloudViewerConfig } from '../../js/functional/state_util'
import { Vector3Type } from '../../js/functional/types'
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
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  const viewerConfigUpdater = new ViewerConfigUpdater()
  Session.subscribe(() => {
    viewerConfigUpdater.updateCamera()
  })
  viewerConfigUpdater.setContainer(document.createElement('div'))

  Session.dispatch(moveCameraAndTarget(
    (new Vector3D()).fromObject({ x: 0, y: 1, z: 0 }),
    (new Vector3D()).fromObject({ x: 0, y: 0, z: 0 })
  ))

  let state = Session.getState()
  let viewerConfig = getCurrentPointCloudViewerConfig(state)

  expectVector3TypesClose(viewerConfig.position, { x: 0, y: 1, z: 0 })
  expectVector3TypesClose(viewerConfig.target, { x: 0, y: 0, z: 0 })

  const mouseDownEvent = new MouseEvent(
    'mousedown',
    { clientX: 0, clientY: 0, button: 2 }
  )
  viewerConfigUpdater.onMouseDown(mouseDownEvent)

  const mouseMoveEvent = new MouseEvent(
    'mousemove',
    {
      clientX: MOUSE_CORRECTION_FACTOR / 2,
      clientY: MOUSE_CORRECTION_FACTOR / 2,
      button: 2
    }
  )
  viewerConfigUpdater.onMouseMove(mouseMoveEvent)

  const mouseUpEvent = new MouseEvent(
    'mouseup',
    {
      clientX: MOUSE_CORRECTION_FACTOR / 2,
      clientY: MOUSE_CORRECTION_FACTOR / 2,
      button: 2
    }
  )
  viewerConfigUpdater.onMouseUp(mouseUpEvent)

  state = Session.getState()
  viewerConfig = getCurrentPointCloudViewerConfig(state)

  expectVector3TypesClose(viewerConfig.position, { x: 1, y: 1, z: 1 })
  expectVector3TypesClose(viewerConfig.target, { x: 1, y: 0, z: 1 })
})
