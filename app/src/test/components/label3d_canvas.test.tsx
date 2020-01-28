import { cleanup, render } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import * as THREE from 'three'
import * as action from '../../js/action/common'
import { moveCamera, moveCameraAndTarget } from '../../js/action/point_cloud'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label3dCanvas } from '../../js/components/label3d_canvas'
import { getCurrentViewerConfig, getShape } from '../../js/functional/state_util'
import { makePointCloudViewerConfig } from '../../js/functional/states'
import { CubeType, PointCloudViewerConfigType } from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import { updateThreeCameraAndRenderer } from '../../js/view_config/point_cloud'
import { testJson } from '../test_point_cloud_objects'
import { expectVector3TypesClose } from '../util'

const canvasId = 0
const width = 1000
const height = 1000

jest.mock('three', () => {
  const three = jest.requireActual('three')
  return {
    ...three,
    WebGLRenderer: class WebGlRenderer {
      constructor (_params: THREE.WebGLRendererParameters) {
        return
      }
      /** Mock render */
      public render (): void {
        return
      }

      /** Mock set size */
      public setSize (): void {
        return
      }
    }
  }
})

beforeEach(() => {
  Session.devMode = false
  initStore(testJson)
  Session.subscribe(() => Session.label3dList.updateState(Session.getState()))
  Session.activeViewerId = canvasId
  Session.pointClouds.length = 0
  Session.pointClouds.push({ [-1]: new THREE.BufferGeometry() })
  Session.dispatch(action.loadItem(0, -1))
})

afterEach(cleanup)

/** Set up component for testing */
function setUpLabel3dCanvas (
  paneId: number = 0
): Label3dCanvas {
  const canvasRef: React.RefObject<Label3dCanvas> = React.createRef()
  Session.dispatch(
    action.addViewerConfig(canvasId, makePointCloudViewerConfig(paneId))
  )

  const camera = new THREE.PerspectiveCamera(45, 1, 1, 1000)

  Session.subscribe(() => {
    const config = Session.getState().user.viewerConfigs[canvasId]
    updateThreeCameraAndRenderer(config as PointCloudViewerConfigType, camera)
  })

  const display = document.createElement('div')
  display.getBoundingClientRect = () => {
    return {
      width,
      height,
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      x: 0,
      y: 0,
      toJSON:  () => {
        return {
          width,
          height,
          top: 0,
          bottom: 0,
          left: 0,
          right: 0,
          x: 0,
          y: 0
        }
      }
    }
  }

  render(
    <div style={{ width: `${width}px`, height: `${height}px` }}>
      <Label3dCanvas
        classes={{
          label3d_canvas: 'label3dcanvas'
        }}
        id={0}
        display={display}
        ref={canvasRef}
        camera={camera}
      />
    </div>
  )

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  expect(canvasRef.current).not.toBeNull()
  expect(canvasRef.current).not.toBeUndefined()

  if (canvasRef.current) {
    return canvasRef.current
  }

  throw new Error('3D canvas ref did not initialize')
}

/** Create mouse down event */
function mouseEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return {
    clientX: x,
    clientY: y,
    stopPropagation: () => { return }
  } as React.MouseEvent<HTMLCanvasElement>
}

test('Add 3d bbox', () => {
  const canvas = setUpLabel3dCanvas()

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  canvas.onKeyDown(spaceEvent)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let cube = getShape(state, 0, 0, 0) as CubeType
  let canvasConfig =
    getCurrentViewerConfig(state, canvasId) as PointCloudViewerConfigType
  expect(canvasConfig).not.toBeNull()
  expectVector3TypesClose(cube.center, canvasConfig.target)
  expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
  expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
  expect(cube.anchorIndex).toEqual(0)

  // Move target randomly a few times and
  // make sure that the bounding box is always created at the target
  const maxVal = 100
  const position = new Vector3D()
  position.fromObject(canvasConfig.position)
  const target = new Vector3D()
  for (let i = 1; i <= 10; i += 1) {
    target[0] = Math.random() * 2 - 1
    target[1] = Math.random() * 2 - 1
    target[2] = Math.random() * 2 - 1
    target.multiplyScalar(maxVal)
    Session.dispatch(moveCameraAndTarget(
      position, target, canvasId, canvasConfig
    ))

    canvas.onKeyDown(spaceEvent)
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(i + 1)
    cube = getShape(state, 0, i, 0) as CubeType
    canvasConfig =
      getCurrentViewerConfig(state, canvasId) as PointCloudViewerConfigType
    expect(canvasConfig).not.toBeNull()

    expectVector3TypesClose(canvasConfig.position, position)
    expectVector3TypesClose(canvasConfig.target, target)
    expectVector3TypesClose(cube.center, canvasConfig.target)
    expectVector3TypesClose(cube.orientation, { x: 0, y: 0, z: 0 })
    expectVector3TypesClose(cube.size, { x: 1, y: 1, z: 1 })
    expect(cube.anchorIndex).toEqual(0)
  }
})

test('Move axis aligned 3d bbox along z axis', () => {
  const canvas = setUpLabel3dCanvas()

  let state = Session.getState()

  let canvasConfig =
    getCurrentViewerConfig(state, canvasId) as PointCloudViewerConfigType

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D(), canvasId, canvasConfig
  ))

  state = Session.getState()
  canvasConfig =
    getCurrentViewerConfig(state, canvasId) as PointCloudViewerConfigType

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  canvas.onKeyDown(spaceEvent)
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  canvas.onKeyDown(tEvent)

  const position = new Vector3D()
  position[1] = 10
  Session.dispatch(moveCamera(
    position,
    canvasId,
    canvasConfig
  ))

  canvas.onMouseMove(mouseEvent(width / 2., height * 9. / 20))
  canvas.onMouseDown(mouseEvent(width / 2., height * 9. / 20))
  canvas.onMouseMove(mouseEvent(width / 2., height / 4.))
  canvas.onMouseUp(mouseEvent(width / 2., height / 4.))

  state = Session.getState()
  let cube = getShape(state, 0, 0, 0) as CubeType
  const center = (new Vector3D()).fromObject(cube.center)
  expect(center[2]).toBeGreaterThan(0)
  expect(center[0]).toBeCloseTo(0)
  expect(center[1]).toBeCloseTo(0)

  canvas.onMouseMove(mouseEvent(width / 2., height / 4.))
  canvas.onMouseDown(mouseEvent(width / 2., height / 4.))
  canvas.onMouseMove(mouseEvent(width / 2., height * 9 / 20))
  canvas.onMouseUp(mouseEvent(width / 2., height * 9 / 20))

  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType
  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
})
