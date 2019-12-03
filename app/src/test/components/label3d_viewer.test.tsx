import { cleanup, render } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import * as THREE from 'three'
import * as action from '../../js/action/common'
import { moveCamera, moveCameraAndTarget } from '../../js/action/point_cloud'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label3dViewer } from '../../js/components/label3d_viewer'
import { getCurrentViewerConfig, getShape } from '../../js/functional/state_util'
import { makePointCloudViewerConfig } from '../../js/functional/states'
import { CubeType, PointCloudViewerConfigType } from '../../js/functional/types'
import { Vector3D } from '../../js/math/vector3d'
import { testJson } from '../test_point_cloud_objects'
import { expectVector3TypesClose } from '../util'

const viewerId = 0
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
  Session.activeViewerId = viewerId
  Session.pointClouds.length = 0
  Session.pointClouds.push({ [-1]: new THREE.Points() })
  Session.dispatch(action.loadItem(0, -1))
})

afterEach(cleanup)

/** Set up component for testing */
function setUpLabel3dViewer (
  paneId: number = 0
): Label3dViewer {
  const viewerRef: React.RefObject<Label3dViewer> = React.createRef()
  Session.dispatch(
    action.addViewerConfig(viewerId, makePointCloudViewerConfig(paneId))
  )

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
      <Label3dViewer
        classes={{
          label3d_canvas: 'label3dcanvas'
        }}
        id={0}
        display={display}
        ref={viewerRef}
      />
    </div>
  )

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  expect(viewerRef.current).not.toBeNull()
  expect(viewerRef.current).not.toBeUndefined()

  if (viewerRef.current) {
    return viewerRef.current
  }

  throw new Error('3D viewer ref did not initialize')
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
  const viewer = setUpLabel3dViewer()

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })

  viewer.onKeyDown(spaceEvent)
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

    viewer.onKeyDown(spaceEvent)
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
  const viewer = setUpLabel3dViewer()

  let state = Session.getState()

  const viewerConfig =
    getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType

  Session.dispatch(moveCameraAndTarget(
    new Vector3D(), new Vector3D(), viewerId, viewerConfig
  ))

  const spaceEvent = new KeyboardEvent('keydown', { key: ' ' })
  viewer.onKeyDown(spaceEvent)
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)

  const labelId = Number(Object.keys(state.task.items[0].labels)[0])
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelId))

  const tEvent = new KeyboardEvent('keydown', { key: 't' })
  viewer.onKeyDown(tEvent)

  const position = new Vector3D()
  position[1] = 10
  Session.dispatch(moveCamera(
    position,
    viewerId,
    viewerConfig
  ))

  viewer.onMouseMove(mouseEvent(width / 2., height * 9. / 20))
  viewer.onMouseDown(mouseEvent(width / 2., height * 9. / 20))
  viewer.onMouseMove(mouseEvent(width / 2., height / 4.))
  viewer.onMouseUp(mouseEvent(width / 2., height / 4.))

  state = Session.getState()
  let cube = getShape(state, 0, 0, 0) as CubeType
  const center = (new Vector3D()).fromObject(cube.center)
  expect(center[2]).toBeGreaterThan(0)
  expect(center[0]).toBeCloseTo(0)
  expect(center[1]).toBeCloseTo(0)

  viewer.onMouseMove(mouseEvent(width / 2., height / 4.))
  viewer.onMouseDown(mouseEvent(width / 2., height / 4.))
  viewer.onMouseMove(mouseEvent(width / 2., height * 9 / 20))
  viewer.onMouseUp(mouseEvent(width / 2., height * 9 / 20))

  state = Session.getState()
  cube = getShape(state, 0, 0, 0) as CubeType
  expectVector3TypesClose(cube.center, { x: 0, y: 0, z: 0 })
})
