import _ from 'lodash'
import * as THREE from 'three'
import * as action from '../../js/action/common'
import { moveCameraAndTarget } from '../../js/action/point_cloud'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label3DList } from '../../js/drawable/label3d_list'
import { getCurrentPointCloudViewerConfig, getShape } from '../../js/functional/state_util'
import {
  CubeType,
  PointCloudViewerConfigType,
  Vector3Type
} from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import { testJson } from '../test_point_cloud_objects'

/**
 * Create ThreeJS Perspective Camera from viewer config fields
 * @param viewerConfig
 */
function getCameraFromViewerConfig (viewerConfig: PointCloudViewerConfigType) {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1
  camera.updateProjectionMatrix()

  camera.up.x = viewerConfig.verticalAxis.x
  camera.up.y = viewerConfig.verticalAxis.y
  camera.up.z = viewerConfig.verticalAxis.z
  camera.position.x = viewerConfig.position.x
  camera.position.y = viewerConfig.position.y
  camera.position.z = viewerConfig.position.z
  const targetPosition = new THREE.Vector3(
    viewerConfig.target.x,
    viewerConfig.target.y,
    viewerConfig.target.z
  )
  camera.lookAt(targetPosition)
  camera.updateProjectionMatrix()
  camera.updateMatrix()
  camera.updateMatrixWorld(true)

  return camera
}

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

test('Draw 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(Session.getState(),
      Session.getState().user.select.item)
  })

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  // Add box w/ initial viewer config
  label3dList.onKeyDown(spaceEvent)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let cube = getShape(state, 0, 0, 0) as CubeType
  let viewerConfig =
    getCurrentPointCloudViewerConfig(state)
  expect(viewerConfig).not.toBeNull()
  expectVector3TypesClose(cube.center, viewerConfig.target)
  expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
  expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
  expect(cube.anchorIndex).toEqual(0)

  // Move target randomly a few times and
  // make sure that the bounding box is always created at the target
  const maxVal = 100
  const position = new Vector3D()
  position.fromObject(viewerConfig.position)
  const target = new Vector3D()
  for (let i = 1; i <= 10; i += 1) {
    target[0] = Math.random() * 2 - 1
    target[1] = Math.random() * 2 - 1
    target[2] = Math.random() * 2 - 1
    target.multiplyScalar(maxVal)
    Session.dispatch(moveCameraAndTarget(position, target))

    label3dList.onKeyDown(spaceEvent)
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(i + 1)
    cube = getShape(state, 0, i, 0) as CubeType
    viewerConfig =
      getCurrentPointCloudViewerConfig(state)
    expect(viewerConfig).not.toBeNull()

    expectVector3TypesClose(viewerConfig.position, position)
    expectVector3TypesClose(viewerConfig.target, target)
    expectVector3TypesClose(cube.center, viewerConfig.target)
    expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
    expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
    expect(cube.anchorIndex).toEqual(0)
  }
})

test('Move 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(Session.getState(),
      Session.getState().user.select.item)
  })
  let state = Session.getState()

  const viewerConfig =
    getCurrentPointCloudViewerConfig(state)
  expectVector3TypesClose(viewerConfig.position, { x: 0, y: 10, z: 0 })
  expectVector3TypesClose(viewerConfig.target, { x: 0, y: 0, z: 0 })
  expectVector3TypesClose(viewerConfig.verticalAxis, { x: 0, y: 0, z: 1 })

  const camera = getCameraFromViewerConfig(viewerConfig)

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  // Add box
  label3dList.onKeyDown(spaceEvent)
  state = Session.getState()
  let cube = getShape(state, 0, 0, 0) as CubeType
  const initialPosition = cube.center
  expectVector3TypesClose(initialPosition, { x: 0, y: 0, z: 0 })

  const raycaster = new THREE.Raycaster()

  // Move mouse to center of screen and highlight box
  label3dList.onMouseMove(0., 0., camera, raycaster)

  // Drag
  label3dList.onMouseDown()
  label3dList.onMouseMove(0.5, 0.5, camera, raycaster)
  label3dList.onMouseUp()

  // Raycast for testing
  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType
  const viewPlaneNormal =
    (new Vector3D()).fromObject(viewerConfig.target).toThree()
  viewPlaneNormal.sub(
    (new Vector3D()).fromObject(viewerConfig.position).toThree())

  raycaster.setFromCamera(new THREE.Vector2(0., 0.), camera)
  const box = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: THREE.FaceColors,
        transparent: true,
        opacity: 0.5
      })
    )
  const intersects = raycaster.intersectObjects([box])
  expect(intersects.length).toBeGreaterThan(0)

  const initialIntersection = intersects[0].point
  const intersectionToCamera = new THREE.Vector3()
  intersectionToCamera.copy(camera.position)
  intersectionToCamera.sub(initialIntersection)

  const projection = new THREE.Vector3(0.5, 0.5, -1)
  projection.unproject(camera)
  projection.sub(camera.position)
  projection.normalize()

  const dist = -intersectionToCamera.dot(viewPlaneNormal) /
    projection.dot(viewPlaneNormal)

  const newProjection = new THREE.Vector3()
  newProjection.copy(projection)
  newProjection.multiplyScalar(dist)

  // Set box position to point
  newProjection.add(camera.position)

  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType

  newProjection.y -= 0.5

  expectVector3TypesClose(newProjection, cube.center)
})

test('Scale 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(Session.getState(),
      Session.getState().user.select.item)
  })
  let state = Session.getState()

  const viewerConfig =
    getCurrentPointCloudViewerConfig(state)
  viewerConfig.position.x = 0
  viewerConfig.position.y = 0
  viewerConfig.position.z = 10

  Session.dispatch(moveCameraAndTarget(
    (new Vector3D()).fromObject(viewerConfig.position),
    (new Vector3D()).fromObject(viewerConfig.target)
  ))

  const camera = getCameraFromViewerConfig(viewerConfig)

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  // Add box
  label3dList.onKeyDown(spaceEvent)

  const raycaster = new THREE.Raycaster()

  // Move mouse to center of screen and highlight box
  label3dList.onMouseMove(0., 0., camera, raycaster)

  const sEvent = new KeyboardEvent('keydown', { key: 's' })
  // Change edit mode
  label3dList.onKeyDown(sEvent)

  // Drag
  label3dList.onMouseDown()
  label3dList.onKeyUp(sEvent)
  label3dList.onMouseMove(0, 0, camera, raycaster)
  label3dList.onMouseUp()

  state = Session.getState()
  const cube = getShape(state, 0, 0, 0) as CubeType

  expectVector3TypesClose(cube.size, { x: 1, y: 0.5, z: 1 })
  expectVector3TypesClose(cube.center, { x: 0, y: -0.25, z: 0 })
})

test('Extrude 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(Session.getState(),
      Session.getState().user.select.item)
  })
  let state = Session.getState()

  const viewerConfig =
    getCurrentPointCloudViewerConfig(state)

  const camera = getCameraFromViewerConfig(viewerConfig)

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  // Add box
  label3dList.onKeyDown(spaceEvent)

  const raycaster = new THREE.Raycaster()

  // Move mouse to center of screen and highlight box
  label3dList.onMouseMove(0., 0., camera, raycaster)

  const sEvent = new KeyboardEvent('keydown', { key: 'e' })
  // Change edit mode
  label3dList.onKeyDown(sEvent)

  // Drag
  label3dList.onMouseDown()
  label3dList.onKeyUp(sEvent)
  label3dList.onMouseMove(0, 0, camera, raycaster)
  label3dList.onMouseUp()

  state = Session.getState()
  const cube = getShape(state, 0, 0, 0) as CubeType

  expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 0.5 })
  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: -0.25 })

})

test('Rotate 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(Session.getState(),
      Session.getState().user.select.item)
  })
  let state = Session.getState()

  const viewerConfig =
    getCurrentPointCloudViewerConfig(state)

  const camera = getCameraFromViewerConfig(viewerConfig)

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  // Add box
  label3dList.onKeyDown(spaceEvent)

  const raycaster = new THREE.Raycaster()

  const maxDistance = 0.15
  // Move mouse to highlight box
  label3dList.onMouseMove(maxDistance, 0., camera, raycaster)

  const sEvent = new KeyboardEvent('keydown', { key: 'e' })
  // Change edit mode
  label3dList.onKeyDown(sEvent)

  state = Session.getState()
  expect(state.user.select.item).toEqual(0)

  // Drag
  label3dList.onMouseDown()
  label3dList.onKeyUp(sEvent)

  // Move mouse in circle around box
  const numSteps = 10
  for (let i = 0; i <= numSteps; i++) {
    const angle = 2. * Math.PI * i / numSteps
    const x = Math.cos(angle) * maxDistance
    const y = Math.sin(angle) * maxDistance
    label3dList.onMouseMove(x, y, camera, raycaster)
  }
  label3dList.onMouseUp()

  state = Session.getState()
  const cube = getShape(state, 0, 0, 0) as CubeType

  // Check that cube is back at starting orientation
  expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })

  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
  expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
})
