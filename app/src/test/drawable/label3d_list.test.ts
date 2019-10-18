import _ from 'lodash'
import * as THREE from 'three'
// import * as THREE from 'three'
import * as action from '../../js/action/common'
import { moveCamera, moveCameraAndTarget } from '../../js/action/point_cloud'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label3DList } from '../../js/drawable/3d/label3d_list'
import { getCurrentPointCloudViewerConfig, getShape } from '../../js/functional/state_util'
import { CubeType, Vector3Type } from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import { updateThreeCameraAndRenderer } from '../../js/view_config/point_cloud'
import { testJson } from '../test_point_cloud_objects'

/**
 * Check equality between two Vector3Type objects
 * @param v1
 * @param v2
 */
function expectVector3TypesClose (v1: Vector3Type, v2: Vector3Type, num = 2) {
  expect(v1.x).toBeCloseTo(v2.x, num)
  expect(v1.y).toBeCloseTo(v2.y, num)
  expect(v1.z).toBeCloseTo(v2.z, num)
}

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

test('Add 3d bbox', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(
      Session.getState(),
      Session.getState().user.select.item
    )
  })

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  label3dList.onKeyDown(spaceEvent)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let cube = getShape(state, 0, 0, 0) as CubeType
  let viewerConfig = getCurrentPointCloudViewerConfig(state)
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
    viewerConfig = getCurrentPointCloudViewerConfig(state)
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
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(
      Session.getState(),
      Session.getState().user.select.item
    )
  })

  let state = Session.getState()

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D()
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dList.onKeyDown(spaceEvent)
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  label3dList.onKeyDown(tEvent)

  const position = new Vector3D()
  position[1] = 10
  Session.dispatch(moveCamera(
    position
  ))

  state = Session.getState()
  const viewerConfig = getCurrentPointCloudViewerConfig(state)
  updateThreeCameraAndRenderer(viewerConfig, camera)
  camera.updateMatrixWorld(true)

  const raycastableShapes = label3dList.getRaycastableShapes()

  const raycaster = new THREE.Raycaster()
  raycaster.near = 1.0
  raycaster.far = 100.0
  raycaster.linePrecision = 0.02

  raycaster.setFromCamera(new THREE.Vector2(0, 0.1), camera)
  let intersections =
    raycaster.intersectObjects(raycastableShapes as unknown as THREE.Object3D[])
  expect(intersections.length).toBeGreaterThan(0)

  label3dList.onMouseMove(0, 0.1, camera, intersections[0])
  label3dList.onMouseDown(0, 0.1, camera)
  label3dList.onMouseMove(0, 0.5, camera)
  label3dList.onMouseUp()

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

  label3dList.onMouseMove(0, 0.5, camera, intersections[0])
  label3dList.onMouseDown(0, 0.5, camera)
  label3dList.onMouseMove(0, 0.1, camera)
  label3dList.onMouseUp()

  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType
  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
})

test('Move axis aligned 3d bbox along all axes', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(
      Session.getState(),
      Session.getState().user.select.item
    )
  })

  let state = Session.getState()

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D()
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dList.onKeyDown(spaceEvent)

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  label3dList.onKeyDown(tEvent)

  // Set camera to each of 6 axis aligned locations around cube
  // 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4= +z, 5 = -z
  for (let camLoc = 0; camLoc < 6; camLoc++) {
    const position = new Vector3D()
    position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
    Session.dispatch(moveCamera(
      position
    ))

    state = Session.getState()
    const viewerConfig = getCurrentPointCloudViewerConfig(state)
    updateThreeCameraAndRenderer(viewerConfig, camera)
    camera.updateMatrixWorld(true)

    const raycastableShapes = label3dList.getRaycastableShapes()

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

      label3dList.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dList.onMouseDown(vecX, vecY, camera)
      label3dList.onMouseMove(5 * vecX, 5 * vecY, camera)
      label3dList.onMouseUp()

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

      label3dList.onMouseMove(5 * vecX, 5 * vecY, camera, intersections[0])
      label3dList.onMouseDown(5 * vecX, 5 * vecY, camera)
      label3dList.onMouseMove(vecX, vecY, camera)
      label3dList.onMouseUp()

      state = Session.getState()
      cube = getShape(state, 0, 0, 0) as CubeType
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
    }
  }
})

test('Scale axis aligned 3d bbox along all axes', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
  camera.aspect = 1

  const label3dList = new Label3DList()
  Session.subscribe(() => {
    label3dList.updateState(
      Session.getState(),
      Session.getState().user.select.item
    )
  })

  let state = Session.getState()

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D()
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  label3dList.onKeyDown(spaceEvent)

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(labelId))

  const sEvent = new KeyboardEvent('keydown', { key: 's' })
  label3dList.onKeyDown(sEvent)

  // Set camera to each of 6 axis aligned locations around cube
  // 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4= +z, 5 = -z
  for (let camLoc = 0; camLoc < 6; camLoc++) {
    const position = new Vector3D()
    position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
    Session.dispatch(moveCamera(
      position
    ))

    state = Session.getState()
    const viewerConfig = getCurrentPointCloudViewerConfig(state)
    updateThreeCameraAndRenderer(viewerConfig, camera)
    camera.updateMatrixWorld(true)

    const raycastableShapes = label3dList.getRaycastableShapes()

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

      label3dList.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dList.onMouseDown(vecX, vecY, camera)
      label3dList.onMouseMove(5 * vecX, 5 * vecY, camera)
      label3dList.onMouseUp()

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

      label3dList.onMouseMove(5 * vecX, 5 * vecY, camera, intersections[0])
      label3dList.onMouseDown(5 * vecX, 5 * vecY, camera)
      label3dList.onMouseMove(vecX, vecY, camera)
      label3dList.onMouseUp()

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
      Session.devMode = false
      initStore(testJson)
      const itemIndex = 0
      Session.dispatch(action.goToItem(itemIndex))

      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
      camera.aspect = 1

      const label3dList = new Label3DList()
      Session.subscribe(() => {
        label3dList.updateState(
          Session.getState(),
          Session.getState().user.select.item
        )
      })

      let state = Session.getState()

      Session.dispatch(moveCameraAndTarget(
        new Vector3D(), new Vector3D()
      ))

      const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
      label3dList.onKeyDown(spaceEvent)

      state = Session.getState()
      expect(_.size(state.task.items[0].labels)).toEqual(1)

      const labelId = Number(Object.keys(state.task.items[0].labels)[0])
      Session.dispatch(selectLabel(labelId))

      const rEvent = new KeyboardEvent('keydown', { key: 'r' })
      label3dList.onKeyDown(rEvent)

      const position = new Vector3D()
      position[Math.floor(camLoc / 2)] = 10 * (camLoc % 1 === 0 ? -1 : 1)
      Session.dispatch(moveCamera(
        position
      ))

      state = Session.getState()
      const viewerConfig = getCurrentPointCloudViewerConfig(state)
      updateThreeCameraAndRenderer(viewerConfig, camera)
      camera.updateMatrixWorld(true)

      const raycastableShapes = label3dList.getRaycastableShapes()

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

      label3dList.onMouseMove(vecX, vecY, camera, intersections[0])
      label3dList.onMouseDown(vecX, vecY, camera)
      label3dList.onMouseMove(2 * vecX, 2 * vecY, camera)
      label3dList.onMouseUp()

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

      label3dList.onMouseMove(2 * vecX, 2 * vecY, camera, intersections[0])
      label3dList.onMouseDown(2 * vecX, 2 * vecY, camera)
      label3dList.onMouseMove(vecX, vecY, camera)
      label3dList.onMouseUp()

      state = Session.getState()
      cube = getShape(state, 0, 0, 0) as CubeType
      expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
      expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 }, 1)
    }
  }
})
