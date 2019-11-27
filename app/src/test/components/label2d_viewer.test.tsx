import { cleanup, render } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dViewer } from '../../js/components/label2d_viewer'
import { getShape } from '../../js/functional/state_util'
import { makeImageViewerConfig } from '../../js/functional/states'
import { PolygonType, RectType } from '../../js/functional/types'
import { testJson } from '../test_image_objects'

const viewerRef: React.RefObject<Label2dViewer> = React.createRef()

beforeEach(() => {
  cleanup()
  initStore(testJson)
})

afterEach(cleanup)

beforeAll(() => {
  Session.devMode = false
  Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  setUpLabel2dViewer(1000, 1000)
})

/** Set up component for testing */
function setUpLabel2dViewer (width: number, height: number) {
  Session.dispatch(action.addViewerConfig(0, makeImageViewerConfig(0)))

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
      <Label2dViewer
        classes={{
          label2d_canvas: 'label2dcanvas',
          control_canvas: 'controlcanvas'
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
}

/** Create mouse down event */
function mouseDownEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mousedown',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

/** Create mouse down event */
function mouseUpEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mouseup',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

/** Create mouse down event */
function mouseMoveEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mousemove',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

test('Draw 2d boxes to label2d list', () => {
  if (viewerRef.current) {
    // Draw first box
    viewerRef.current.onMouseMove(mouseMoveEvent(1, 1))
    viewerRef.current.onMouseDown(mouseDownEvent(1, 1))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseUp(mouseUpEvent(50, 50))
    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    let rect = getShape(state, 0, 0, 0) as RectType
    expect(rect.x1).toEqual(1)
    expect(rect.y1).toEqual(1)
    expect(rect.x2).toEqual(50)
    expect(rect.y2).toEqual(50)

    // Second box
    viewerRef.current.onMouseMove(mouseMoveEvent(25, 20))
    viewerRef.current.onMouseDown(mouseDownEvent(25, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(15, 15))
    viewerRef.current.onMouseMove(mouseMoveEvent(70, 85))
    viewerRef.current.onMouseUp(mouseUpEvent(70, 85))

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(2)
    rect = getShape(state, 0, 1, 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(20)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(85)

    // third box
    viewerRef.current.onMouseMove(mouseMoveEvent(15, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(15, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(23, 24))
    viewerRef.current.onMouseMove(mouseMoveEvent(60, 70))
    viewerRef.current.onMouseUp(mouseUpEvent(60, 70))
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, 2, 0) as RectType
    expect(rect.x1).toEqual(15)
    expect(rect.y1).toEqual(10)
    expect(rect.x2).toEqual(60)
    expect(rect.y2).toEqual(70)

    // resize the second box
    viewerRef.current.onMouseMove(mouseMoveEvent(25, 20))
    viewerRef.current.onMouseDown(mouseDownEvent(25, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(15, 18))
    viewerRef.current.onMouseMove(mouseMoveEvent(30 ,34))
    viewerRef.current.onMouseUp(mouseUpEvent(30, 34))
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, 1, 0) as RectType
    expect(rect.x1).toEqual(30)
    expect(rect.y1).toEqual(34)

    // flip top left and bottom right corner
    viewerRef.current.onMouseMove(mouseMoveEvent(30, 34))
    viewerRef.current.onMouseDown(mouseDownEvent(30, 34))
    viewerRef.current.onMouseMove(mouseMoveEvent(90, 90))
    viewerRef.current.onMouseUp(mouseUpEvent(90, 90))
    state = Session.getState()
    rect = getShape(state, 0, 1, 0) as RectType
    expect(rect.x1).toEqual(70)
    expect(rect.y1).toEqual(85)
    expect(rect.x2).toEqual(90)
    expect(rect.y2).toEqual(90)

    // move
    viewerRef.current.onMouseMove(mouseMoveEvent(30, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(30, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(40, 15))
    viewerRef.current.onMouseUp(mouseUpEvent(40, 15))
    state = Session.getState()
    rect = getShape(state, 0, 2, 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(15)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(75)
  }
})

test('Draw 2d polygons to label2d list', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw the first polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))

    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))

    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    /**
     * drawing the first polygon
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */
    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(0)

    // drag when drawing
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 0))

    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     */
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    let polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(4)
    expect(polygon.points[0].x).toEqual(10)
    expect(polygon.points[0].y).toEqual(10)
    expect(polygon.points[0].type).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(100)
    expect(polygon.points[1].y).toEqual(100)
    expect(polygon.points[1].type).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(200)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].type).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(100)
    expect(polygon.points[3].y).toEqual(0)
    expect(polygon.points[3].type).toEqual('vertex')

    // draw second polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))

    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 400))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 400))

    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseDown(mouseDownEvent(700, 700))
    viewerRef.current.onMouseUp(mouseUpEvent(700, 700))

    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(2)
    polygon = getShape(state, 0, 1, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(500)
    expect(polygon.points[0].y).toEqual(500)
    expect(polygon.points[0].type).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(600)
    expect(polygon.points[1].y).toEqual(400)
    expect(polygon.points[1].type).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(700)
    expect(polygon.points[2].y).toEqual(700)
    expect(polygon.points[2].type).toEqual('vertex')
    expect(polygon.points.length).toEqual(3)
    // expect(Session.viewerRef.current.labelList.length).toEqual(2)
  }
})

test('2d polygons highlighted and selected', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw first polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseDown(mouseDownEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))

    viewerRef.current.onMouseMove(mouseMoveEvent(210, 210))
    viewerRef.current.onMouseMove(mouseMoveEvent(210, 210))
    viewerRef.current.onMouseDown(mouseDownEvent(210, 210))
    viewerRef.current.onMouseUp(mouseUpEvent(210, 210))

    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseDown(mouseDownEvent(310, 260))
    viewerRef.current.onMouseUp(mouseUpEvent(310, 260))

    viewerRef.current.onMouseMove(mouseMoveEvent(410, 210))
    viewerRef.current.onMouseMove(mouseMoveEvent(410, 210))
    viewerRef.current.onMouseDown(mouseDownEvent(410, 210))
    viewerRef.current.onMouseUp(mouseUpEvent(410, 210))

    viewerRef.current.onMouseMove(mouseMoveEvent(210, 110))
    viewerRef.current.onMouseMove(mouseMoveEvent(210, 110))
    viewerRef.current.onMouseDown(mouseDownEvent(210, 110))
    viewerRef.current.onMouseUp(mouseUpEvent(210, 110))

    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseDown(mouseDownEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */
    let selected = Session.label2dList.selectedLabels
    expect(selected[0].labelId).toEqual(0)

    // draw second polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))

    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 400))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 400))

    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseDown(mouseDownEvent(700, 700))
    viewerRef.current.onMouseUp(mouseUpEvent(700, 700))

    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // change selected label
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseDown(mouseDownEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(140, 140))
    viewerRef.current.onMouseMove(mouseMoveEvent(140, 140))
    viewerRef.current.onMouseUp(mouseUpEvent(140, 140))
    selected = Session.label2dList.selectedLabels
    expect(selected[0].labelId).toEqual(0)
  }
})

test('validation check for polygon2d', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw a valid polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseDown(mouseDownEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))

    viewerRef.current.onMouseMove(mouseMoveEvent(210, 210))
    viewerRef.current.onMouseMove(mouseMoveEvent(210, 210))
    viewerRef.current.onMouseDown(mouseDownEvent(210, 210))
    viewerRef.current.onMouseUp(mouseUpEvent(210, 210))

    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseDown(mouseDownEvent(310, 260))
    viewerRef.current.onMouseUp(mouseUpEvent(310, 260))

    viewerRef.current.onMouseMove(mouseMoveEvent(410, 210))
    viewerRef.current.onMouseMove(mouseMoveEvent(410, 210))
    viewerRef.current.onMouseDown(mouseDownEvent(410, 210))
    viewerRef.current.onMouseUp(mouseUpEvent(410, 210))

    viewerRef.current.onMouseMove(mouseMoveEvent(210, 110))
    viewerRef.current.onMouseMove(mouseMoveEvent(210, 110))
    viewerRef.current.onMouseDown(mouseDownEvent(210, 110))
    viewerRef.current.onMouseUp(mouseUpEvent(210, 110))

    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseDown(mouseDownEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))

    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */

    // draw one invalid polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))

    viewerRef.current.onMouseMove(mouseMoveEvent(400, 300))
    viewerRef.current.onMouseMove(mouseMoveEvent(400, 300))
    viewerRef.current.onMouseDown(mouseDownEvent(400, 300))
    viewerRef.current.onMouseUp(mouseUpEvent(400, 300))

    viewerRef.current.onMouseMove(mouseMoveEvent(300, 200))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 200))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 200))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 200))

    viewerRef.current.onMouseMove(mouseMoveEvent(300, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 0))

    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))

    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     * polygon 2: (200, 100) (400, 300) (300, 200) (300, 0) invalid
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(Session.label2dList.labelList.length).toEqual(1)

    // drag the polygon to an invalid shape
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 260))
    viewerRef.current.onMouseDown(mouseDownEvent(310, 260))
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(310, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(310, 0))

    /**
     * polygon 1: (120, 120) (210, 210) (310, 0) (410, 210) (210, 110)
     * polygon 1 is an invalid shape
     */

    state = Session.getState()
    const polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[2].x).toEqual(310)
    expect(polygon.points[2].y).toEqual(260)
    expect(polygon.points[2].type).toEqual('vertex')
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */

    // draw a too small polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(0, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(0, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(0, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(0, 0))

    viewerRef.current.onMouseMove(mouseMoveEvent(1, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(1, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(1, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(1, 0))

    viewerRef.current.onMouseMove(mouseMoveEvent(0, 1))
    viewerRef.current.onMouseMove(mouseMoveEvent(0, 1))
    viewerRef.current.onMouseDown(mouseDownEvent(0, 1))
    viewerRef.current.onMouseUp(mouseUpEvent(0, 1))

    viewerRef.current.onMouseMove(mouseMoveEvent(0, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(0, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(0, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(0, 0))

    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     * polygon 2: (0, 0) (1, 0) (0, 1) too small, invalid
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(Session.label2dList.labelList.length).toEqual(1)
  }
})

test('2d polygons drag vertices, midpoints and edges', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     */

    // drag a vertex
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 100))
    let state = Session.getState()
    let polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[2].x).toEqual(300)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].type).toEqual('vertex')
    /**
     * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
     */

    // drag midpoints
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 150))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 150))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 150))
    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[2].x).toEqual(200)
    expect(polygon.points[2].y).toEqual(150)
    expect(polygon.points[2].type).toEqual('vertex')
    expect(polygon.points.length).toEqual(5)
    /**
     * polygon 1: (10, 10) (100, 100) (200, 150) (300, 100) (100, 0)
     */

    // drag edges
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseDown(mouseDownEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))
    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    expect(polygon.points[0].type).toEqual('vertex')
    expect(polygon.points.length).toEqual(5)
    /**
     * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
     */
  }
})

test('2d polygons delete vertex and draw bezier curve', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw a polygon and delete vertex when drawing
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 0))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 0))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 0))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(350, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(350, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 200))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 200))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 200))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 200))
    viewerRef.current.onMouseMove(mouseMoveEvent(320, 130))
    viewerRef.current.onMouseMove(mouseMoveEvent(320, 130))
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
    viewerRef.current.onMouseDown(mouseDownEvent(320, 130))
    viewerRef.current.onMouseUp(mouseUpEvent(320, 130))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 150))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 150))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 150))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 150))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 100))
    /**
     * polygon: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(Session.label2dList.labelList.length).toEqual(1)

    let polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(5)
    expect(polygon.points[0].x).toEqual(250)
    expect(polygon.points[0].y).toEqual(100)
    expect(polygon.points[0].type).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(300)
    expect(polygon.points[1].y).toEqual(0)
    expect(polygon.points[1].type).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(350)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].type).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(320)
    expect(polygon.points[3].y).toEqual(130)
    expect(polygon.points[3].type).toEqual('vertex')
    expect(polygon.points[4].x).toEqual(300)
    expect(polygon.points[4].y).toEqual(150)
    expect(polygon.points[4].type).toEqual('vertex')

    // delete vertex when closed
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(275, 125))
    viewerRef.current.onMouseMove(mouseMoveEvent(275, 125))
    viewerRef.current.onMouseDown(mouseDownEvent(275, 125))
    viewerRef.current.onMouseUp(mouseUpEvent(2750, 1250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 150))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 150))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 150))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 150))
    viewerRef.current.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
    /**
     * polygon: (250, 100) (300, 0) (350, 100) (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(4)
    expect(polygon.points[3].x).toEqual(320)
    expect(polygon.points[3].y).toEqual(130)
    expect(polygon.points[3].type).toEqual('vertex')

    // draw bezier curve
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'c' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(335, 115))
    viewerRef.current.onMouseMove(mouseMoveEvent(335, 115))
    viewerRef.current.onMouseDown(mouseDownEvent(335, 115))
    viewerRef.current.onMouseUp(mouseUpEvent(335, 115))
    viewerRef.current.onKeyUp(new KeyboardEvent('keyup', { key: 'c' }))
    /**
     * polygon: (250, 100) (300, 0) (350, 100)
     *          [ (340, 110) (330, 120) <bezier curve control points>]
     *          (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(6)
    expect(polygon.points[3].x).toEqual(340)
    expect(polygon.points[3].y).toEqual(110)
    expect(polygon.points[3].type).toEqual('bezier')
    expect(polygon.points[4].x).toEqual(330)
    expect(polygon.points[4].y).toEqual(120)
    expect(polygon.points[4].type).toEqual('bezier')

    // drag bezier curve control points
    viewerRef.current.onMouseMove(mouseMoveEvent(340, 110))
    viewerRef.current.onMouseMove(mouseMoveEvent(340, 110))
    viewerRef.current.onMouseDown(mouseDownEvent(340, 110))
    viewerRef.current.onMouseMove(mouseMoveEvent(340, 90))
    viewerRef.current.onMouseMove(mouseMoveEvent(340, 90))
    viewerRef.current.onMouseUp(mouseUpEvent(340, 90))
    /**
     * polygon: (250, 100) (300, 0) (350, 100)
     *          [ (340, 90) (330, 120) <bezier curve control points>]
     *          (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(6)
    expect(polygon.points[2].x).toEqual(350)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].type).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(340)
    expect(polygon.points[3].y).toEqual(90)
    expect(polygon.points[3].type).toEqual('bezier')
    expect(polygon.points[4].x).toEqual(330)
    expect(polygon.points[4].y).toEqual(120)
    expect(polygon.points[4].type).toEqual('bezier')

    // delete vertex on bezier curve
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(350, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(350, 100))
    viewerRef.current.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
    /**
     * polygon: (250, 100) (300, 0) (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points.length).toEqual(3)
  }
})

test('2d polygons multi-select and multi-label moving', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw first polygon
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */

    // draw second polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 400))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseDown(mouseDownEvent(700, 700))
    viewerRef.current.onMouseUp(mouseUpEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // draw third polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 350))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 350))
    viewerRef.current.onMouseDown(mouseDownEvent(350, 350))
    viewerRef.current.onMouseUp(mouseUpEvent(350, 350))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

     // select label 1
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 600))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 600))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 600))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 600))

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(1)
    expect(state.user.select.labels[0][0]).toEqual(1)
    expect(Session.label2dList.selectedLabels.length).toEqual(1)
    expect(Session.label2dList.selectedLabels[0].labelId).toEqual(1)

    // select label 1, 2, 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseDown(mouseDownEvent(50, 50))
    viewerRef.current.onMouseUp(mouseUpEvent(50, 50))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(3)
    expect(Session.label2dList.selectedLabels.length).toEqual(3)

    // unselect label 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(0, 0))

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(2)
    expect(Session.label2dList.selectedLabels.length).toEqual(2)
    expect(Session.label2dList.labelList[2].highlighted).toEqual(false)

    // select three labels
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(3)
    expect(Session.label2dList.selectedLabels.length).toEqual(3)

    // move
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseDown(mouseDownEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(60, 60))
    viewerRef.current.onMouseMove(mouseMoveEvent(60, 60))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))
    /**
     * polygon 1: (110, 110) (200, 200) (300, 200)
     * polygon 2: (600, 600) (700, 500) (800, 800)
     * polygon 3: (350, 350) (400, 350) (450, 450)
     */

    state = Session.getState()
    let polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    polygon = getShape(state, 0, 1, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(600)
    expect(polygon.points[0].y).toEqual(600)
    polygon = getShape(state, 0, 2, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(350)
    expect(polygon.points[0].y).toEqual(350)
  }
})

test('2d polygons linking labels and moving', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw first polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(10, 10))
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */

    // draw second polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 400))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseDown(mouseDownEvent(700, 700))
    viewerRef.current.onMouseUp(mouseUpEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // draw third polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 350))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 350))
    viewerRef.current.onMouseDown(mouseDownEvent(350, 350))
    viewerRef.current.onMouseUp(mouseUpEvent(350, 350))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

    // select label 2 and 0
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 300))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 300))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 300))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 300))
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(2)
    expect(state.user.select.labels[0][0]).toEqual(2)
    expect(state.user.select.labels[0][1]).toEqual(0)
    expect(Session.label2dList.selectedLabels.length).toEqual(2)
    expect(Session.label2dList.selectedLabels[0].labelId).toEqual(2)
    expect(Session.label2dList.selectedLabels[1].labelId).toEqual(0)

    // select label 1 and 2
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 600))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 600))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 600))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 600))
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseDown(mouseDownEvent(50, 50))
    viewerRef.current.onMouseUp(mouseUpEvent(50, 50))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    // link label 1 and 2
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     * group 1: 1, 2
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(4)
    expect(_.size(Session.label2dList.labelList)).toEqual(3)
    expect(Session.label2dList.labelList[0].color).toEqual(
    Session.label2dList.labelList[1].color
  )

    // reselect label 1 and 2
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseMove(mouseMoveEvent(50, 50))
    viewerRef.current.onMouseDown(mouseDownEvent(50, 50))
    viewerRef.current.onMouseUp(mouseUpEvent(50, 50))

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(2)
    expect(Session.label2dList.selectedLabels.length).toEqual(2)

    // moving group 1
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(20, 20))
    viewerRef.current.onMouseDown(mouseDownEvent(20, 20))
    viewerRef.current.onMouseMove(mouseMoveEvent(60, 60))
    viewerRef.current.onMouseMove(mouseMoveEvent(60, 60))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseMove(mouseMoveEvent(120, 120))
    viewerRef.current.onMouseUp(mouseUpEvent(120, 120))
    /**
     * polygon 1: (110, 110) (200, 200) (300, 200)
     * polygon 2: (600, 600) (700, 500) (800, 800)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     * group 1: 1, 2
     */

    state = Session.getState()
    let polygon = getShape(state, 0, 0, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    polygon = getShape(state, 0, 1, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(600)
    expect(polygon.points[0].y).toEqual(600)
    polygon = getShape(state, 0, 2, 0) as PolygonType
    expect(polygon.points[0].x).toEqual(250)
    expect(polygon.points[0].y).toEqual(250)
  }
})

test('2d polygons unlinking', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (viewerRef.current) {
    // draw first polygon
    viewerRef.current.onMouseDown(mouseDownEvent(10, 10))
    viewerRef.current.onMouseUp(mouseUpEvent(10, 10))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onMouseMove(mouseMoveEvent(200, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(200, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(200, 100))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */

    // draw second polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    viewerRef.current.onMouseMove(mouseMoveEvent(600, 400))
    viewerRef.current.onMouseDown(mouseDownEvent(600, 400))
    viewerRef.current.onMouseUp(mouseUpEvent(600, 400))
    viewerRef.current.onMouseMove(mouseMoveEvent(700, 700))
    viewerRef.current.onMouseDown(mouseDownEvent(700, 700))
    viewerRef.current.onMouseUp(mouseUpEvent(700, 700))
    viewerRef.current.onMouseMove(mouseMoveEvent(500, 500))
    viewerRef.current.onMouseDown(mouseDownEvent(500, 500))
    viewerRef.current.onMouseUp(mouseUpEvent(500, 500))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // draw third polygon
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(300, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(300, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(300, 250))
    viewerRef.current.onMouseMove(mouseMoveEvent(350, 350))
    viewerRef.current.onMouseDown(mouseDownEvent(350, 350))
    viewerRef.current.onMouseUp(mouseUpEvent(350, 350))
    viewerRef.current.onMouseMove(mouseMoveEvent(250, 250))
    viewerRef.current.onMouseDown(mouseDownEvent(250, 250))
    viewerRef.current.onMouseUp(mouseUpEvent(250, 250))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

    // select polygon 1 and 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    // link polygon 1 and 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(4)
    expect(_.size(Session.label2dList.labelList)).toEqual(3)
    expect(Session.label2dList.labelList[0].color).toEqual(
      Session.label2dList.labelList[2].color
  )
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     * group 1: polygon 1, 3
     */

    // select polygon 1, 2, 3
    viewerRef.current.onMouseMove(mouseMoveEvent(550, 550))
    viewerRef.current.onMouseDown(mouseDownEvent(550, 550))
    viewerRef.current.onMouseUp(mouseUpEvent(550, 550))
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    // unlink polygon 1 and 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'L' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'L' }))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    expect(_.size(Session.label2dList.labelList)).toEqual(3)

    // unselect polygon 1
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
    viewerRef.current.onMouseMove(mouseMoveEvent(100, 100))
    viewerRef.current.onMouseDown(mouseDownEvent(100, 100))
    viewerRef.current.onMouseUp(mouseUpEvent(100, 100))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

    // link polygon 2 and 3
    viewerRef.current.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
    viewerRef.current.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     * group 1: polygon 2, 3
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(4)
    expect(_.size(Session.label2dList.labelList)).toEqual(3)
    expect(Session.label2dList.labelList[1].color).toEqual(
      Session.label2dList.labelList[2].color)
  }
})
