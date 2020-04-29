import { cleanup } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { getShape } from '../../js/functional/state_util'
import { IdType, PolygonType, RectType } from '../../js/functional/types'
import { testJson } from '../test_image_objects'
import { findNewLabelsFromState } from '../util'
import { drawPolygon, keyDown, keyUp, mouseDown, mouseMove, mouseMoveClick, mouseUp, setUpLabel2dCanvas } from './label2d_canvas_util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

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
  setUpLabel2dCanvas(Session.dispatch.bind(Session), canvasRef, 1000, 1000)
})

test('Draw 2d boxes to label2d list', () => {
  if (canvasRef.current) {
    const labelIds: IdType[] = []
    const label2d = canvasRef.current
    // Draw first box
    mouseMove(label2d, 1, 1)
    mouseDown(label2d, 1, 1)
    mouseMove(label2d, 50, 50)
    mouseUp(label2d, 50, 50)
    let state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    let rect = getShape(state, 0, labelIds[0], 0) as RectType
    expect(rect.x1).toEqual(1)
    expect(rect.y1).toEqual(1)
    expect(rect.x2).toEqual(50)
    expect(rect.y2).toEqual(50)

    // Second box
    mouseMove(label2d, 25, 20)
    mouseDown(label2d, 25, 20)
    mouseMove(label2d, 15, 15)
    mouseMove(label2d, 70, 85)
    mouseUp(label2d, 70, 85)

    state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(2)
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(20)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(85)

    // third box
    mouseMove(label2d, 15, 10)
    mouseDown(label2d, 15, 10)
    mouseMove(label2d, 23, 24)
    mouseMove(label2d, 60, 70)
    mouseUp(label2d, 60, 70)
    state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, labelIds[2], 0) as RectType
    expect(rect.x1).toEqual(15)
    expect(rect.y1).toEqual(10)
    expect(rect.x2).toEqual(60)
    expect(rect.y2).toEqual(70)

    // resize the second box
    mouseMove(label2d, 25, 20)
    mouseDown(label2d, 25, 20)
    mouseMove(label2d, 15, 18)
    mouseMove(label2d, 30, 34)
    mouseUp(label2d, 30, 34)
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(30)
    expect(rect.y1).toEqual(34)

    // flip top left and bottom right corner
    mouseMove(label2d, 30, 34)
    mouseDown(label2d, 30, 34)
    mouseMove(label2d, 90, 90)
    mouseUp(label2d, 90, 90)
    state = Session.getState()
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(70)
    expect(rect.y1).toEqual(85)
    expect(rect.x2).toEqual(90)
    expect(rect.y2).toEqual(90)

    // move
    mouseMove(label2d, 30, 10)
    mouseDown(label2d, 30, 10)
    mouseMove(label2d, 40, 15)
    mouseUp(label2d, 40, 15)
    state = Session.getState()
    rect = getShape(state, 0, labelIds[2], 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(15)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(75)
  }
})

test('Draw 2d polygons to label2d list', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw the first polygon
    drawPolygon(label2d, [[10, 10], [100, 100], [200, 100]])
    /**
     * drawing the first polygon
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */
    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(0)

    // drag when drawing
    mouseMove(label2d, 200, 10)
    mouseMove(label2d, 200, 10)
    mouseDown(label2d, 200, 10)
    mouseMove(label2d, 100, 0)
    mouseMove(label2d, 100, 0)
    mouseDown(label2d, 100, 0)
    mouseUp(label2d, 100, 0)

    mouseMove(label2d, 10, 10)
    mouseMoveClick(label2d, 10, 10)
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     */
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(4)
    expect(polygon.points[0].x).toEqual(10)
    expect(polygon.points[0].y).toEqual(10)
    expect(polygon.points[0].pointType).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(100)
    expect(polygon.points[1].y).toEqual(100)
    expect(polygon.points[1].pointType).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(200)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].pointType).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(100)
    expect(polygon.points[3].y).toEqual(0)
    expect(polygon.points[3].pointType).toEqual('vertex')

    // draw second polygon
    drawPolygon(label2d, [[500, 500], [600, 400], [700, 700]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(2)
    polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(500)
    expect(polygon.points[0].y).toEqual(500)
    expect(polygon.points[0].pointType).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(600)
    expect(polygon.points[1].y).toEqual(400)
    expect(polygon.points[1].pointType).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(700)
    expect(polygon.points[2].y).toEqual(700)
    expect(polygon.points[2].pointType).toEqual('vertex')
    expect(polygon.points.length).toEqual(3)
    // expect(Session.canvasRef.current.labelList.length).toEqual(2)
  }
})

test('2d polygons highlighted and selected', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw first polygon
    drawPolygon(
      label2d, [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])
    let selected = Session.label2dList.selectedLabels
    expect(selected[0].labelId).toEqual(labelIds[0])

    // draw second polygon
    drawPolygon(label2d, [[500, 500], [600, 400], [700, 700]])
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // change selected label
    mouseMove(label2d, 120, 120)
    mouseMove(label2d, 120, 120)
    mouseDown(label2d, 120, 120)
    mouseMove(label2d, 140, 140)
    mouseMove(label2d, 140, 140)
    mouseUp(label2d, 140, 140)
    selected = Session.label2dList.selectedLabels
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])
    expect(selected[0].labelId).toEqual(labelIds[0])
  }
})

test('validation check for polygon2d', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw a valid polygon
    drawPolygon(label2d,
      [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */

    // draw one invalid polygon
    drawPolygon(label2d, [[200, 100], [400, 300], [300, 200], [300, 0]])

    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     * polygon 2: (200, 100) (400, 300) (300, 200) (300, 0) invalid
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(Session.label2dList.labelList.length).toEqual(1)

    // drag the polygon to an invalid shape
    mouseMove(label2d, 310, 260)
    mouseMove(label2d, 310, 260)
    mouseDown(label2d, 310, 260)
    mouseMove(label2d, 310, 0)
    mouseMove(label2d, 310, 0)
    mouseUp(label2d, 310, 0)

    /**
     * polygon 1: (120, 120) (210, 210) (310, 0) (410, 210) (210, 110)
     * polygon 1 is an invalid shape
     */

    state = Session.getState()
    const polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[2].x).toEqual(310)
    expect(polygon.points[2].y).toEqual(260)
    expect(polygon.points[2].pointType).toEqual('vertex')
    /**
     * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
     */

    // draw a too small polygon
    drawPolygon(label2d, [[0, 0], [1, 0], [0, 1]])

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
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    drawPolygon(label2d, [[10, 10], [100, 100], [200, 100], [100, 0]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    // drag a vertex
    mouseMove(label2d, 200, 100)
    mouseMove(label2d, 200, 100)
    mouseDown(label2d, 200, 100)
    mouseMove(label2d, 300, 100)
    mouseMove(label2d, 300, 100)
    mouseUp(label2d, 300, 100)
    let state = Session.getState()
    let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[2].x).toEqual(300)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].pointType).toEqual('vertex')
    /**
     * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
     */

    // drag midpoints
    mouseMove(label2d, 200, 100)
    mouseMove(label2d, 200, 100)
    mouseDown(label2d, 200, 100)
    mouseMove(label2d, 200, 150)
    mouseMove(label2d, 200, 150)
    mouseUp(label2d, 200, 150)
    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[2].x).toEqual(200)
    expect(polygon.points[2].y).toEqual(150)
    expect(polygon.points[2].pointType).toEqual('vertex')
    expect(polygon.points.length).toEqual(5)
    /**
     * polygon 1: (10, 10) (100, 100) (200, 150) (300, 100) (100, 0)
     */

    // drag edges
    mouseMove(label2d, 20, 20)
    mouseMove(label2d, 20, 20)
    mouseDown(label2d, 20, 20)
    mouseMove(label2d, 120, 120)
    mouseMove(label2d, 120, 120)
    mouseUp(label2d, 120, 120)
    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    expect(polygon.points[0].pointType).toEqual('vertex')
    expect(polygon.points.length).toEqual(5)
    /**
     * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
     */
  }
})

test('2d polygons delete vertex and draw bezier curve', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw a polygon and delete vertex when drawing
    mouseMoveClick(label2d, 200, 100)
    keyDown(label2d, 'd')
    keyUp(label2d, 'd')
    mouseMoveClick(label2d, 250, 100)
    mouseMoveClick(label2d, 300, 0)
    mouseMoveClick(label2d, 350, 100)
    mouseMoveClick(label2d, 300, 200)
    mouseMove(label2d, 320, 130)
    mouseMove(label2d, 320, 130)
    keyDown(label2d, 'd')
    keyUp(label2d, 'd')
    mouseDown(label2d, 320, 130)
    mouseUp(label2d, 320, 130)
    mouseMoveClick(label2d, 300, 150)
    mouseMoveClick(label2d, 250, 100)

    /**
     * polygon: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(Session.label2dList.labelList.length).toEqual(1)

    let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(5)
    expect(polygon.points[0].x).toEqual(250)
    expect(polygon.points[0].y).toEqual(100)
    expect(polygon.points[0].pointType).toEqual('vertex')
    expect(polygon.points[1].x).toEqual(300)
    expect(polygon.points[1].y).toEqual(0)
    expect(polygon.points[1].pointType).toEqual('vertex')
    expect(polygon.points[2].x).toEqual(350)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].pointType).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(320)
    expect(polygon.points[3].y).toEqual(130)
    expect(polygon.points[3].pointType).toEqual('vertex')
    expect(polygon.points[4].x).toEqual(300)
    expect(polygon.points[4].y).toEqual(150)
    expect(polygon.points[4].pointType).toEqual('vertex')

    // delete vertex when closed
    keyDown(label2d, 'd')
    mouseMove(label2d, 275, 125)
    mouseMove(label2d, 275, 125)
    mouseDown(label2d, 275, 125)
    mouseUp(label2d, 2750, 1250)
    mouseMove(label2d, 300, 150)
    mouseMoveClick(label2d, 300, 150)
    keyUp(label2d, 'd')
    /**
     * polygon: (250, 100) (300, 0) (350, 100) (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(4)
    expect(polygon.points[3].x).toEqual(320)
    expect(polygon.points[3].y).toEqual(130)
    expect(polygon.points[3].pointType).toEqual('vertex')

    // draw bezier curve
    keyDown(label2d, 'c')
    mouseMove(label2d, 335, 115)
    mouseMoveClick(label2d, 335, 115)
    keyUp(label2d, 'c')
    /**
     * polygon: (250, 100) (300, 0) (350, 100)
     *          [ (340, 110) (330, 120) <bezier curve control points>]
     *          (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(6)
    expect(polygon.points[3].x).toEqual(340)
    expect(polygon.points[3].y).toEqual(110)
    expect(polygon.points[3].pointType).toEqual('bezier')
    expect(polygon.points[4].x).toEqual(330)
    expect(polygon.points[4].y).toEqual(120)
    expect(polygon.points[4].pointType).toEqual('bezier')

    // drag bezier curve control points
    mouseMove(label2d, 340, 110)
    mouseMove(label2d, 340, 110)
    mouseDown(label2d, 340, 110)
    mouseMove(label2d, 340, 90)
    mouseMove(label2d, 340, 90)
    mouseUp(label2d, 340, 90)
    /**
     * polygon: (250, 100) (300, 0) (350, 100)
     *          [ (340, 90) (330, 120) <bezier curve control points>]
     *          (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(6)
    expect(polygon.points[2].x).toEqual(350)
    expect(polygon.points[2].y).toEqual(100)
    expect(polygon.points[2].pointType).toEqual('vertex')
    expect(polygon.points[3].x).toEqual(340)
    expect(polygon.points[3].y).toEqual(90)
    expect(polygon.points[3].pointType).toEqual('bezier')
    expect(polygon.points[4].x).toEqual(330)
    expect(polygon.points[4].y).toEqual(120)
    expect(polygon.points[4].pointType).toEqual('bezier')

    // delete vertex on bezier curve
    keyDown(label2d, 'd')
    mouseMoveClick(label2d, 350, 100)
    keyUp(label2d, 'd')
    /**
     * polygon: (250, 100) (300, 0) (320, 130)
     */

    state = Session.getState()
    polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points.length).toEqual(3)
  }
})

test('2d polygons multi-select and multi-label moving', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw first polygon
    drawPolygon(label2d, [[10, 10], [100, 100], [200, 100]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    // draw second polygon
    drawPolygon(label2d, [[500, 500], [600, 400], [700, 700]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    // draw third polygon
    drawPolygon(label2d, [[250, 250], [300, 250], [350, 350]])

    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

    // select label 1
    mouseMoveClick(label2d, 600, 600)

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(1)
    expect(state.user.select.labels[0][0]).toEqual(1)
    expect(Session.label2dList.selectedLabels.length).toEqual(1)
    expect(Session.label2dList.selectedLabels[0].labelId).toEqual(1)

    // select label 1, 2, 3
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 300, 250)
    mouseMoveClick(label2d, 50, 50)
    keyUp(label2d, 'Meta')

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(3)
    expect(Session.label2dList.selectedLabels.length).toEqual(3)

    // unselect label 3
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 300, 250)
    keyUp(label2d, 'Meta')
    mouseMove(label2d, 0, 0)

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(2)
    expect(Session.label2dList.selectedLabels.length).toEqual(2)
    expect(Session.label2dList.labelList[2].highlighted).toEqual(false)

    // select three labels
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 300, 250)
    keyUp(label2d, 'Meta')

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(3)
    expect(Session.label2dList.selectedLabels.length).toEqual(3)

    // move
    mouseMove(label2d, 20, 20)
    mouseMove(label2d, 20, 20)
    mouseDown(label2d, 20, 20)
    mouseMove(label2d, 60, 60)
    mouseMove(label2d, 60, 60)
    mouseMove(label2d, 120, 120)
    mouseMove(label2d, 120, 120)
    mouseUp(label2d, 120, 120)
    /**
     * polygon 1: (110, 110) (200, 200) (300, 200)
     * polygon 2: (600, 600) (700, 500) (800, 800)
     * polygon 3: (350, 350) (400, 350) (450, 450)
     */

    state = Session.getState()
    let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(600)
    expect(polygon.points[0].y).toEqual(600)
    polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(350)
    expect(polygon.points[0].y).toEqual(350)
  }
})

test('2d polygons linking labels and moving', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw first polygon
    drawPolygon(label2d, [[10, 10], [100, 100], [200, 100]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    // draw second polygon
    drawPolygon(label2d, [[500, 500], [600, 400], [700, 700]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    // draw third polygon
    drawPolygon(label2d, [[250, 250], [300, 250], [350, 350]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

    // select label 2 and 0
    mouseMoveClick(label2d, 300, 300)
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 100, 100)
    keyUp(label2d, 'Meta')
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
    mouseMoveClick(label2d, 600, 600)
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 50, 50)
    keyUp(label2d, 'Meta')

    // link label 1 and 2
    keyDown(label2d, 'l')
    keyUp(label2d, 'l')
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
    mouseMoveClick(label2d, 300, 250)
    mouseMoveClick(label2d, 50, 50)

    state = Session.getState()
    expect(state.user.select.labels[0].length).toEqual(2)
    expect(Session.label2dList.selectedLabels.length).toEqual(2)

    // moving group 1
    mouseMove(label2d, 20, 20)
    mouseMove(label2d, 20, 20)
    mouseDown(label2d, 20, 20)
    mouseMove(label2d, 60, 60)
    mouseMove(label2d, 60, 60)
    mouseMove(label2d, 120, 120)
    mouseMove(label2d, 120, 120)
    mouseUp(label2d, 120, 120)
    /**
     * polygon 1: (110, 110) (200, 200) (300, 200)
     * polygon 2: (600, 600) (700, 500) (800, 800)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     * group 1: 1, 2
     */

    state = Session.getState()
    let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(110)
    expect(polygon.points[0].y).toEqual(110)
    polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(600)
    expect(polygon.points[0].y).toEqual(600)
    polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
    expect(polygon.points[0].x).toEqual(250)
    expect(polygon.points[0].y).toEqual(250)
  }
})

test('2d polygons unlinking', () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  if (canvasRef.current) {
    const label2d = canvasRef.current
    // draw first polygon
    drawPolygon(label2d, [[10, 10], [100, 100], [200, 100]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     */

    // draw second polygon
    drawPolygon(label2d, [[500, 500], [600, 400], [700, 700]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     */

    // draw third polygon
    drawPolygon(label2d, [[250, 250], [300, 250], [350, 350]])
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    let state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)

    // select polygon 1 and 3
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 100, 100)
    keyUp(label2d, 'Meta')

    // link polygon 1 and 3
    keyDown(label2d, 'l')
    keyUp(label2d, 'l')

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
    mouseMoveClick(label2d, 550, 550)
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 100, 100)
    keyUp(label2d, 'Meta')

    // unlink polygon 1 and 3
    keyDown(label2d, 'L')
    keyUp(label2d, 'L')
    /**
     * polygon 1: (10, 10) (100, 100) (200, 100)
     * polygon 2: (500, 500) (600, 400) (700, 700)
     * polygon 3: (250, 250) (300, 250) (350, 350)
     */

    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    expect(_.size(Session.label2dList.labelList)).toEqual(3)

    // unselect polygon 1
    keyDown(label2d, 'Meta')
    mouseMoveClick(label2d, 100, 100)
    keyUp(label2d, 'Meta')

    // link polygon 2 and 3
    keyDown(label2d, 'l')
    keyUp(label2d, 'l')
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
