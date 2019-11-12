import _ from 'lodash'
import * as THREE from 'three'
// import * as THREE from 'three'
import * as action from '../../js/action/common'
import { moveCamera, moveCameraAndTarget } from '../../js/action/point_cloud'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label3DHandler } from '../../js/drawable/3d/label3d_handler'
import { getCurrentViewerConfig, getShape } from '../../js/functional/state_util'
import { makePointCloudViewerConfig } from '../../js/functional/states'
import { CubeType, PointCloudViewerConfigType } from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import { updateThreeCameraAndRenderer } from '../../js/view_config/point_cloud'
import { testJson } from '../test_point_cloud_objects'
import { expectVector3TypesClose } from '../util'

/**
 * Get active axis given camLoc and axis
 * @param camLoc
 * @param axis
 */
function getActiveAxis (camLoc: number, axis: number) {
  if (Math.floor(camLoc / 2) === 0) {
    if (axis <= 1) {
      return 2
    } else if (axis >= 2) {
      return 1
    }
  } else if (Math.floor(camLoc / 2) === 1) {
    if (axis <= 1) {
      return 2
    } else if (axis >= 2) {
      return 0
    }
  } else if (Math.floor(camLoc / 2) === 2) {
    if (axis <= 1) {
      return 0
    } else if (axis >= 2) {
      return 1
    }
  }
}

/**
 * Get active axis for rotation given camLoc and axis
 * @param camLoc
 * @param axis
 */
function getActiveAxisForRotation (camLoc: number, axis: number) {
  if (Math.floor(camLoc / 2) === 0) {
    if (axis <= 1) {
      return 1
    } else if (axis >= 2) {
      return 2
    }
  } else if (Math.floor(camLoc / 2) === 1) {
    if (axis <= 1) {
      return 0
    } else if (axis >= 2) {
      return 2
    }
  } else if (Math.floor(camLoc / 2) === 2) {
    if (axis <= 1) {
      return 1
    } else if (axis >= 2) {
      return 0
    }
  }
}

/**
 * Initialize Session, label 3d list, label 3d handler
 */
function initializeTestingObjects (): [Label3DHandler, number] {
  Session.devMode = false
  initStore(testJson)
  Session.dispatch(action.addViewerConfig(1, makePointCloudViewerConfig(-1)))
  const viewerId = 1

  const label3dHandler = new Label3DHandler()
  Session.subscribe(() => {
    const state = Session.getState()
    Session.label3dList.updateState(state)
    label3dHandler.updateState(state, state.user.select.item, viewerId)
  })

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  return [label3dHandler, viewerId]
}

test('Add 3d bbox', () => {
  const [label3dHandler, viewerId] = initializeTestingObjects()
  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  label3dHandler.onKeyDown(spaceEvent)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let cube = getShape(state, 0, 0, 0) as CubeType
  let viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
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
    Session.dispatch(moveCameraAndTarget(
      position, target, viewerId, viewerConfig
    ))

    label3dHandler.onKeyDown(spaceEvent)
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(i + 1)
    cube = getShape(state, 0, i, 0) as CubeType
    viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
    expect(viewerConfig).not.toBeNull()

    expectVector3TypesClose(viewerConfig.position, position)
    expectVector3TypesClose(viewerConfig.target, target)
    expectVector3TypesClose(cube.center, viewerConfig.target)
    expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
    expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
    expect(cube.anchorIndex).toEqual(0)
  }
})

test('Move axis aligned 3d bbox along z axis', () => {
  const [label3dHandler, viewerId] = initializeTestingObjects()

  let state = Session.getState()

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  let viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D(), viewerId, viewerConfig
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dHandler.onKeyDown(spaceEvent)
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  label3dHandler.onKeyDown(tEvent)

  const position = new Vector3D()
  position[1] = 10
  Session.dispatch(moveCamera(
    position,
    viewerId,
    viewerConfig
  ))

  state = Session.getState()
  viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
  updateThreeCameraAndRenderer(viewerConfig, camera)
  camera.updateMatrixWorld(true)

  const raycastableShapes = Session.label3dList.raycastableShapes

  const raycaster = new THREE.Raycaster()
  raycaster.near = 1.0
  raycaster.far = 100.0
  raycaster.linePrecision = 0.02

  raycaster.setFromCamera(new THREE.Vector2(0, 0.1), camera)
  let intersections =
    raycaster.intersectObjects(raycastableShapes as unknown as THREE.Object3D[])
  expect(intersections.length).toBeGreaterThan(0)

  label3dHandler.onMouseMove(0, 0.1, camera, intersections[0])
  label3dHandler.onMouseDown(0, 0.1, camera)
  label3dHandler.onMouseMove(0, 0.5, camera)
  label3dHandler.onMouseUp()

  state = Session.getState()
  let cube = getShape(state, 0, 0, 0) as CubeType
  const center = (new Vector3D()).fromObject(cube.center)
  expect(center[2]).toBeGreaterThan(0)
  expect(center[0]).toBeCloseTo(0)
  expect(center[1]).toBeCloseTo(0)

  raycaster.setFromCamera(new THREE.Vector2(0, 0.5), camera)
  intersections =
    raycaster.intersectObjects(raycastableShapes as unknown as THREE.Object3D[])
  expect(intersections.length).toBeGreaterThan(0)

  label3dHandler.onMouseMove(0, 0.5, camera, intersections[0])
  label3dHandler.onMouseDown(0, 0.5, camera)
  label3dHandler.onMouseMove(0, 0.1, camera)
  label3dHandler.onMouseUp()

  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType
  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
})

test('Move axis aligned 3d bbox along all axes', () => {
  const [label3dHandler, viewerId] = initializeTestingObjects()

  let state = Session.getState()
  let viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D(), viewerId, viewerConfig
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dHandler.onKeyDown(spaceEvent)

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  label3dHandler.onKeyDown(tEvent)

  // Set camera to each of 6 axis aligned locations around cube
  // 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4= +z, 5 = -z
  for (let camLoc = 0; camLoc < 6; camLoc++) {
    state = Session.getState()
    viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
    const position = new Vector3D()
    position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
    Session.dispatch(moveCamera(
      position, viewerId, viewerConfig
    ))

    state = Session.getState()
    viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
    updateThreeCameraAndRenderer(viewerConfig, camera)
    camera.updateMatrixWorld(true)

    const raycastableShapes = Session.label3dList.raycastableShapes

    const raycaster = new THREE.Raycaster()
    raycaster.near = 1.0
    raycaster.far = 100.0
    raycaster.linePrecision = 0.02

    // From each axis aligned view there is vertical and horizontal axis.
    // Try positive and negative directions. 0 = +v, 1 = -v, 2 = +h, 3 = -h
    for (let axis = 0; axis < 4; axis++) {
      const neg: boolean = (axis % 2 === 1)
      // Because of the view orientation, a positive movement could be along
      // a negative axis. This corrects that for testing.
      const negAxis: boolean = axis >= 2 && (camLoc === 0 || camLoc === 1)
      const vecX = .1 * (axis >= 2 ? 1 : 0) * (neg ? -1 : 1)
      const vecY = .1 * (axis <= 1 ? 1 : 0) * (neg ? -1 : 1)

      raycaster.setFromCamera(new THREE.Vector2(vecX, vecY), camera)
      let intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dHandler.onMouseDown(vecX, vecY, camera)
      label3dHandler.onMouseMove(5 * vecX, 5 * vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      let cube = getShape(state, 0, 0, 0) as CubeType
      const center = (new Vector3D()).fromObject(cube.center)

      // get ActiveAxis based on view point and vertical or horizontal
      const activeAxis = getActiveAxis(camLoc, axis)
      for (let i = 0; i < 3; i++) {
        if (i !== activeAxis) {
          // Check other directions have not moved due to translation
          expect(center[i]).toBeCloseTo(0)
        } else {
          // Check translated direction, accounting for negations
          const pos = (neg ? -1 : 1) * (negAxis ? -1 : 1) * center[i]
          expect(pos).toBeGreaterThan(0)
        }
      }

      raycaster.setFromCamera(new THREE.Vector2(5 * vecX, 5 * vecY), camera)
      intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(5 * vecX, 5 * vecY, camera, intersections[0])
      label3dHandler.onMouseDown(5 * vecX, 5 * vecY, camera)
      label3dHandler.onMouseMove(vecX, vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      cube = getShape(state, 0, 0, 0) as CubeType
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
    }
  }
})

test('Scale axis aligned 3d bbox along all axes', () => {
  const [label3dHandler, viewerId] = initializeTestingObjects()

  let state = Session.getState()
  let viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D(), viewerId, viewerConfig
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dHandler.onKeyDown(spaceEvent)

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

  const sEvent = new KeyboardEvent('keydown', { key: 'e' })
  label3dHandler.onKeyDown(sEvent)

  // Set camera to each of 6 axis aligned locations around cube
  // 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4= +z, 5 = -z
  for (let camLoc = 0; camLoc < 6; camLoc++) {
    state = Session.getState()
    viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
    const position = new Vector3D()
    position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
    Session.dispatch(moveCamera(
      position, viewerId, viewerConfig
    ))

    state = Session.getState()
    viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
    updateThreeCameraAndRenderer(viewerConfig, camera)
    camera.updateMatrixWorld(true)

    const raycastableShapes = Session.label3dList.raycastableShapes

    const raycaster = new THREE.Raycaster()
    raycaster.near = 1.0
    raycaster.far = 100.0
    raycaster.linePrecision = 0.02

    // From each axis aligned view there is vertical and horizontal axis.
    // Try positive and negative directions. 0 = +v, 1 = -v, 2 = +h, 3 = -h
    for (let axis = 0; axis < 4; axis++) {
      const neg: boolean = (axis % 2 === 1)
      // Because of the view orientation, a positive movement could be along
      // a negative axis. This corrects that for testing.
      const negAxis: boolean = axis >= 2 && (camLoc === 0 || camLoc === 1)
      const vecX = .1 * (axis >= 2 ? 1 : 0) * (neg ? -1 : 1)
      const vecY = .1 * (axis <= 1 ? 1 : 0) * (neg ? -1 : 1)

      raycaster.setFromCamera(new THREE.Vector2(vecX, vecY), camera)
      let intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dHandler.onMouseDown(vecX, vecY, camera)
      label3dHandler.onMouseMove(5 * vecX, 5 * vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      let cube = getShape(state, 0, 0, 0) as CubeType

      const center = (new Vector3D()).fromObject(cube.center)
      const size = (new Vector3D()).fromObject(cube.size)

      // get ActiveAxis based on view point and vertical or horizontal
      const activeAxis = getActiveAxis(camLoc, axis)
      // expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })

      for (let i = 0; i < 3; i++) {
        if (i !== activeAxis) {
          // Check other directions have not changed due to scaling
          expect(center[i]).toBeCloseTo(0)
          expect(size[i]).toBeCloseTo(1)
        } else {
          // Check scaling direction, accounting for negations
          const pos = (neg ? -1 : 1) * (negAxis ? -1 : 1) * center[i]
          expect(pos).toBeGreaterThan(0)
          expect(size[i]).toBeGreaterThan(1)
        }
      }

      raycaster.setFromCamera(new THREE.Vector2(5 * vecX, 5 * vecY), camera)
      intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(5 * vecX, 5 * vecY, camera, intersections[0])
      label3dHandler.onMouseDown(5 * vecX, 5 * vecY, camera)
      label3dHandler.onMouseMove(vecX, vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      cube = getShape(state, 0, 0, 0) as CubeType
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
      expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
    }
  }
})

test('Rotate axis aligned 3d bbox around all axes', () => {
  // Set camera to each of 6 axis aligned locations around cube
  // 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4= +z, 5 = -z
  for (let camLoc = 0; camLoc < 6; camLoc++) {
    // From each axis aligned view there is vertical and horizontal axis.
    // Try positive and negative directions. 0 = +v, 1 = -v, 2 = +h, 3 = -h
    for (let axis = 0; axis < 4; axis++) {
      const [label3dHandler, viewerId] = initializeTestingObjects()

      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
      camera.aspect = 1

      let state = Session.getState()
      let viewerConfig =
        getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

      Session.dispatch(moveCameraAndTarget(
        new Vector3D(), new Vector3D(), viewerId, viewerConfig
      ))

      const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
      label3dHandler.onKeyDown(spaceEvent)

      state = Session.getState()
      expect(_.size(state.task.items[0].labels)).toEqual(1)

      const labelId = Number(Object.keys(state.task.items[0].labels)[0])
      Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

      const rEvent = new KeyboardEvent('keydown', { key: 'r' })
      label3dHandler.onKeyDown(rEvent)

      state = Session.getState()
      viewerConfig =
        getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

      const position = new Vector3D()
      position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
      Session.dispatch(moveCamera(
        position, viewerId, viewerConfig
      ))

      state = Session.getState()
      viewerConfig =
        getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
      updateThreeCameraAndRenderer(viewerConfig, camera)
      camera.updateMatrixWorld(true)

      const raycastableShapes = Session.label3dList.raycastableShapes

      const raycaster = new THREE.Raycaster()
      raycaster.near = 1.0
      raycaster.far = 100.0
      raycaster.linePrecision = 0.02

      const neg: boolean = (axis % 2 === 1)
      // Because of the view orientation, a positive movement could be along
      // a negative axis. This corrects that for testing.
      const negAxis: boolean = axis <= 1 && (camLoc >= 2)
      const vecX = .1 * (axis >= 2 ? 1 : 0) * (neg ? -1 : 1)
      const vecY = .1 * (axis <= 1 ? 1 : 0) * (neg ? -1 : 1)

      raycaster.setFromCamera(new THREE.Vector2(vecX, vecY), camera)
      let intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dHandler.onMouseDown(vecX, vecY, camera)
      label3dHandler.onMouseMove(2 * vecX, 2 * vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      let cube = getShape(state, 0, 0, 0) as CubeType
      const orientation = (new Vector3D()).fromObject(cube.orientation)

      // get ActiveAxis based on view point and vertical or horizontal
      const activeAxis = getActiveAxisForRotation(camLoc, axis)
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
      expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })

      for (let i = 0; i < 3; i++) {
        if (i !== activeAxis) {
          // Check other orientations have not moved due to rotation
          expect(orientation[i]).toBeCloseTo(0)
        } else {
          // Check rotated orientations, accounting for negations
          const pos = (neg ? -1 : 1) * (negAxis ? -1 : 1) * orientation[i]
          expect(pos).toBeGreaterThan(0)
        }
      }

      raycaster.setFromCamera(new THREE.Vector2(2 * vecX, 2 * vecY), camera)
      intersections = raycaster.intersectObjects(
        raycastableShapes as unknown as THREE.Object3D[])
      expect(intersections.length).toBeGreaterThan(0)

      label3dHandler.onMouseMove(2 * vecX, 2 * vecY, camera, intersections[0])
      label3dHandler.onMouseDown(2 * vecX, 2 * vecY, camera)
      label3dHandler.onMouseMove(vecX, vecY, camera)
      label3dHandler.onMouseUp()

      state = Session.getState()
      cube = getShape(state, 0, 0, 0) as CubeType
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
      expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 }, 1)
    }
  }
})
