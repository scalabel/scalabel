import { createCanvas } from 'canvas'
import _ from 'lodash'
import * as action from '../../js/action/common'
import { addPolygon2dLabel } from '../../js/action/polygon2d'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Key, ShapeTypeName } from '../../js/common/types'
import { Label2DHandler } from '../../js/drawable/2d/label2d_handler'
import { PathPoint2D, PointType } from '../../js/drawable/2d/path_point2d'
import { getShape } from '../../js/functional/state_util'
import { makeImageViewerConfig } from '../../js/functional/states'
import { IdType, Point2DType, PolygonType, RectType } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { testJson } from '../test_image_objects'
import { findNewLabels } from '../util'

/**
 * Initialize Session, label 3d list, label 3d handler
 */
function initializeTestingObjects (): [Label2DHandler, number] {
  Session.devMode = false
  initStore(testJson)
  Session.dispatch(action.addViewerConfig(1, makeImageViewerConfig(0)))
  const viewerId = 1

  const label2dHandler = new Label2DHandler()
  Session.subscribe(() => {
    const state = Session.getState()
    Session.label2dList.updateState(state)
    label2dHandler.updateState(state)
  })

  Session.dispatch(action.loadItem(0, -1))

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  return [label2dHandler, viewerId]
}

/**
 * Create a polygon based on the input vertices
 * @param label2dHandler
 * @param canvasSize
 * @param points
 */
function drawPolygon (
  label2dHandler: Label2DHandler, canvasSize: Size2D, points: number[][]) {
  let v: Vector2D
  for (const p of points) {
    v = new Vector2D(p[0], p[1])
    label2dHandler.onMouseMove(v, canvasSize, -1, 0)
    label2dHandler.onMouseDown(v, -1, 0)
    label2dHandler.onMouseUp(v, -1, 0)
  }
  v = new Vector2D(points[0][0], points[0][1])
  label2dHandler.onMouseMove(v, canvasSize, -1, 1)
  label2dHandler.onMouseDown(v, -1, 1)
  label2dHandler.onMouseUp(v, -1, 1)
}

test('Draw 2d boxes to label2d list', () => {
  const [label2dHandler] = initializeTestingObjects()

  const labelIds: IdType[] = []
  // Draw first box
  const canvasSize = new Size2D(100, 100)
  label2dHandler.onMouseMove(new Vector2D(1, 1), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(1, 1), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.push(findNewLabels(state.task.items[0].labels, labelIds)[0])
  let rect = getShape(state, 0, labelIds[0], 0) as RectType
  expect(rect.x1).toEqual(1)
  expect(rect.y1).toEqual(1)
  expect(rect.x2).toEqual(10)
  expect(rect.y2).toEqual(10)

  // Second box
  label2dHandler.onMouseMove(new Vector2D(19, 20), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(19, 20), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(25, 25), canvasSize, -1, 0)
  label2dHandler.onMouseMove(new Vector2D(30, 29), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(30, 29), -1, 0)

  state = Session.getState()
  // Make sure a new label is added
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  // Find the new label
  labelIds.push(findNewLabels(state.task.items[0].labels, labelIds)[0])
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(19)
  expect(rect.y1).toEqual(20)
  expect(rect.x2).toEqual(30)
  expect(rect.y2).toEqual(29)

  // third box
  label2dHandler.onMouseMove(new Vector2D(4, 5), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(4, 5), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(15, 15), canvasSize, -1, 0)
  label2dHandler.onMouseMove(new Vector2D(23, 24), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(23, 24), -1, 0)
  state = Session.getState()
  // Make sure a new label is added
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  labelIds.push(findNewLabels(state.task.items[0].labels, labelIds)[0])
  rect = getShape(state, 0, labelIds[2], 0) as RectType
  expect(rect.x1).toEqual(4)
  expect(rect.y1).toEqual(5)
  expect(rect.x2).toEqual(23)
  expect(rect.y2).toEqual(24)

  // resize the second box
  label2dHandler.onMouseMove(new Vector2D(19, 20), canvasSize, 1, 1)
  label2dHandler.onMouseDown(new Vector2D(19, 20), 1, 1)
  label2dHandler.onMouseMove(new Vector2D(15, 18), canvasSize, -1, 0)
  label2dHandler.onMouseMove(new Vector2D(16, 17), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(16, 17), -1, 0)
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(16)
  expect(rect.y1).toEqual(17)

  // flip top left and bottom right corner
  label2dHandler.onMouseMove(new Vector2D(16, 17), canvasSize, 1, 1)
  label2dHandler.onMouseDown(new Vector2D(16, 17), 1, 1)
  label2dHandler.onMouseMove(new Vector2D(42, 43), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(40, 41), -1, 0)
  state = Session.getState()
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(30)
  expect(rect.y1).toEqual(29)
  expect(rect.x2).toEqual(42)
  expect(rect.y2).toEqual(43)

  // move
  label2dHandler.onMouseMove(new Vector2D(32, 31), canvasSize, 1, 0)
  label2dHandler.onMouseDown(new Vector2D(32, 31), 1, 0)
  label2dHandler.onMouseMove(new Vector2D(36, 32), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(36, 32), -1, 0)
  state = Session.getState()
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(34)
  expect(rect.y1).toEqual(30)
  expect(rect.x2).toEqual(46)
  expect(rect.y2).toEqual(44)

  // delete label
  Session.dispatch(action.deleteLabel(0, labelIds[1]))
  expect(Session.label2dList.labelList.length).toEqual(2)
  expect(Session.label2dList.labelList[0].index).toEqual(0)
  expect(Session.label2dList.labelList[0].labelId).toEqual(labelIds[0])
  expect(Session.label2dList.labelList[1].index).toEqual(1)
  expect(Session.label2dList.labelList[1].labelId).toEqual(labelIds[2])
})

test('Draw 2d polygons to label2d list', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  // draw the first polygon
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 0)
  /**
   * drawing the first polygon
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  let state = Session.getState()
  const labelIds: IdType[] = []
  expect(_.size(state.task.items[0].labels)).toEqual(0)

  // drag when drawing
  label2dHandler.onMouseMove(new Vector2D(200, 10), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 10), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(100, 0), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(100, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.push(findNewLabels(state.task.items[0].labels, labelIds)[0])
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
  label2dHandler.onMouseMove(new Vector2D(500, 500), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(500, 500), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(500, 500), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(600, 400), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(600, 400), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(600, 400), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(700, 700), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(700, 700), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(700, 700), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(500, 500), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(500, 500), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(500, 500), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  labelIds.push(findNewLabels(state.task.items[0].labels, labelIds)[0])
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
  expect(Session.label2dList.labelList.length).toEqual(2)
})

test('2d polygons highlighted and selected', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  const labelIds: IdType[] = []

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  let selected = Session.label2dList.selectedLabels
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  expect(selected[0].labelId).toEqual(labelIds[0])

  // draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  // change highlighted label
  label2dHandler.onMouseMove(new Vector2D(130, 130), canvasSize, 0, 0)
  let highlighted = label2dHandler.highlightedLabel
  selected = Session.label2dList.selectedLabels
  expect(Session.label2dList.labelList.length).toEqual(2)
  expect(Session.label2dList.labelList[1].labelId).toEqual(labelIds[1])
  if (!highlighted) {
    throw new Error('no highlightedLabel')
  } else {
    expect(highlighted.labelId).toEqual(labelIds[0])
  }
  expect(selected[0].labelId).toEqual(labelIds[1])

  // change selected label
  label2dHandler.onMouseDown(new Vector2D(130, 130), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(140, 140), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(140, 140), 0, 0)
  highlighted = label2dHandler.highlightedLabel
  selected = Session.label2dList.selectedLabels
  if (highlighted) {
    expect(highlighted.labelId).toEqual(labelIds[0])
  }
  expect(selected[0].labelId).toEqual(labelIds[0])
})

test('validation check for polygon2d', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // draw a valid polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  // draw one invalid polygon
  drawPolygon(label2dHandler, canvasSize,
    [[200, 100], [400, 300], [300, 200], [300, 0]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (200, 100) (400, 300) (300, 200) (300, 0) invalid
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  // drag the polygon to an invalid shape
  label2dHandler.onMouseMove(new Vector2D(310, 260), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(310, 260), 0, 5)
  label2dHandler.onMouseMove(new Vector2D(310, 0), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(310, 0), 0, 5)
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
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, [])[0])

  // draw a too small polygon
  drawPolygon(label2dHandler, canvasSize, [[0, 0], [1, 0], [0, 1]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (0, 0) (1, 0) (0, 1) too small, invalid
   */
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)
})

test('2d polygons drag vertices, midpoints and edges', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // draw a polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[10, 10], [100, 100], [200, 100], [100, 0]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  // drag a vertex
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 5)
  label2dHandler.onMouseMove(new Vector2D(300, 100), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(300, 100), 0, 5)
  let state = Session.getState()
  let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[2].x).toEqual(300)
  expect(polygon.points[2].y).toEqual(100)
  expect(polygon.points[2].pointType).toEqual('vertex')
  /**
   * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
   */

  // drag midpoints
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 4)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 4)
  label2dHandler.onMouseMove(new Vector2D(200, 150), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(200, 150), 0, 5)
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
  label2dHandler.onMouseMove(new Vector2D(20, 20), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(20, 20), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), 0, 0)
  state = Session.getState()
  polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0].x).toEqual(110)
  expect(polygon.points[0].y).toEqual(110)
  expect(polygon.points[0].pointType).toEqual('vertex')
  expect(polygon.points.length).toEqual(5)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test('2d polygons delete vertex and draw bezier curve', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // draw a polygon and delete vertex when drawing
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[250, 100], [300, 0], [350, 100], [320, 130], [300, 150]])
  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
   */
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

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
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(275, 125), canvasSize, 0, 10)
  label2dHandler.onMouseDown(new Vector2D(275, 125), 0, 10)
  label2dHandler.onMouseUp(new Vector2D(275, 125), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 150), canvasSize, 0, 9)
  label2dHandler.onMouseDown(new Vector2D(300, 150), 0, 9)
  label2dHandler.onMouseUp(new Vector2D(300, 150), 0, 9)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
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
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'c' }))
  label2dHandler.onMouseMove(new Vector2D(335, 115), canvasSize, 0, 6)
  label2dHandler.onMouseDown(new Vector2D(335, 125), 0, 6)
  label2dHandler.onMouseUp(new Vector2D(335, 115), 0, 0)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'c' }))
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
  label2dHandler.onMouseMove(new Vector2D(340, 110), canvasSize, 0, 6)
  label2dHandler.onMouseDown(new Vector2D(340, 110), 0, 6)
  label2dHandler.onMouseMove(new Vector2D(340, 90), canvasSize, 0, 6)
  label2dHandler.onMouseUp(new Vector2D(340, 90), 0, 6)
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
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(350, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(350, 100), 0, 5)
  label2dHandler.onMouseUp(new Vector2D(350, 100), 0, 5)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  /**
   * polygon: (250, 100) (300, 0) (320, 130)
   */

  state = Session.getState()
  polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points.length).toEqual(3)
})

test('2d polygons multi-select and multi-label moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  // draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */
  labelIds.push(findNewLabels(
      Session.getState().task.items[0].labels, labelIds)[0])

    // draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

   // select label 1
  label2dHandler.onMouseMove(new Vector2D(600, 600), canvasSize, 1, 0)
  label2dHandler.onMouseDown(new Vector2D(600, 600), 1, 0)
  label2dHandler.onMouseUp(new Vector2D(600, 600), 1, 0)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(1)
  expect(state.user.select.labels[0][0]).toEqual(labelIds[1])
  expect(Session.label2dList.selectedLabels.length).toEqual(1)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(labelIds[1])

  // select label 1, 2, 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, 2, 2)
  label2dHandler.onMouseDown(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseUp(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseMove(new Vector2D(50, 50), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(50, 50), 0, 0)
  label2dHandler.onMouseUp(new Vector2D(50, 50), 0, 0)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // unselect label 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, 2, 2)
  label2dHandler.onMouseDown(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseUp(new Vector2D(300, 250), 2, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)

  // select three labels
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, 2, 2)
  label2dHandler.onMouseDown(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseUp(new Vector2D(300, 250), 2, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // move
  label2dHandler.onMouseMove(new Vector2D(20, 20), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(20, 20), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(60, 60), canvasSize, 0, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), 0, 0)
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
})

test('2d polygons linking labels and moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])

  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

  // draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // select label 2 and 0
  label2dHandler.onMouseMove(new Vector2D(300, 300), canvasSize, 2, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 300), 2, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 300), 2, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, 0, 2)
  label2dHandler.onMouseDown(new Vector2D(100, 100), 0, 2)
  label2dHandler.onMouseUp(new Vector2D(100, 100), 0, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(state.user.select.labels[0][0]).toEqual(labelIds[2])
  expect(state.user.select.labels[0][1]).toEqual(labelIds[0])
  expect(Session.label2dList.selectedLabels.length).toEqual(2)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(labelIds[2])
  expect(Session.label2dList.selectedLabels[1].labelId).toEqual(labelIds[0])

  // select label 1 and 2
  label2dHandler.onMouseMove(new Vector2D(600, 600), canvasSize, 1, 0)
  label2dHandler.onMouseDown(new Vector2D(600, 600), 1, 0)
  label2dHandler.onMouseUp(new Vector2D(600, 600), 1, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(50, 50), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(50, 50), 0, 0)
  label2dHandler.onMouseUp(new Vector2D(50, 50), 0, 0)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  // link label 1 and 2
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))
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
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, 2, 2)
  label2dHandler.onMouseDown(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseUp(new Vector2D(300, 250), 2, 2)
  label2dHandler.onMouseMove(new Vector2D(50, 50), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(50, 50), 0, 0)
  label2dHandler.onMouseUp(new Vector2D(50, 50), 0, 0)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)

  // moving group 1
  label2dHandler.onMouseMove(new Vector2D(20, 20), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(20, 20), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(60, 60), canvasSize, 0, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), 0, 0)
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

  // reshape for one label in group
  label2dHandler.onMouseMove(new Vector2D(110, 110), canvasSize, 0, 1)
  label2dHandler.onMouseDown(new Vector2D(110, 110), 0, 1)
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, 0, 1)
  label2dHandler.onMouseUp(new Vector2D(100, 100), 0, 1)
  /**
   * polygon 1: (100, 100) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = Session.getState()
  polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0].x).toEqual(100)
  expect(polygon.points[0].y).toEqual(100)
  polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polygon.points[0].x).toEqual(600)
  expect(polygon.points[0].y).toEqual(600)
  polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
  expect(polygon.points[0].x).toEqual(250)
  expect(polygon.points[0].y).toEqual(250)
})

test('2d polygons unlinking', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

  // draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // select polygon 1 and 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, 0, 2)
  label2dHandler.onMouseDown(new Vector2D(100, 100), 0, 2)
  label2dHandler.onMouseUp(new Vector2D(100, 100), 0, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  // link polygon 1 and 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))

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
  label2dHandler.onMouseMove(new Vector2D(550, 550), canvasSize, 1, 0)
  label2dHandler.onMouseDown(new Vector2D(550, 550), 1, 0)
  label2dHandler.onMouseUp(new Vector2D(550, 550), 1, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, 0, 2)
  label2dHandler.onMouseDown(new Vector2D(100, 100), 0, 2)
  label2dHandler.onMouseUp(new Vector2D(100, 100), 0, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  // unlink polygon 1 and 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'L' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'L' }))
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)

  // unselect polygon 1
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'Meta' }))
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, 0, 2)
  label2dHandler.onMouseDown(new Vector2D(100, 100), 0, 2)
  label2dHandler.onMouseUp(new Vector2D(100, 100), 0, 2)
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'Meta' }))

  // link polygon 2 and 3
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'l' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keydown', { key: 'l' }))
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
})

test('2d polyline creating', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // draw first polyline
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   */

  // draw second polyline
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   * polyline 2: (500, 500) (600, 400) (700, 700)
   */

  // draw third polyline
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   * polyline 2: (500, 500) (600, 400) (700, 700)
   * polyline 3: (250, 250) (300, 250) (350, 350)
   */

  const state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  let polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(3)
  expect(polyline.points[0].x).toEqual(10)
  expect(polyline.points[0].y).toEqual(10)
  expect(polyline.points[0].pointType).toEqual('vertex')
  expect(polyline.points[1].x).toEqual(100)
  expect(polyline.points[1].y).toEqual(100)
  expect(polyline.points[1].pointType).toEqual('vertex')
  expect(polyline.points[2].x).toEqual(200)
  expect(polyline.points[2].y).toEqual(100)
  expect(polyline.points[2].pointType).toEqual('vertex')

  polyline = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polyline.points.length).toEqual(3)
  expect(polyline.points[0].x).toEqual(500)
  expect(polyline.points[0].y).toEqual(500)
  expect(polyline.points[0].pointType).toEqual('vertex')

  polyline = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polyline.points.length).toEqual(3)
  expect(polyline.points[0].x).toEqual(500)
  expect(polyline.points[0].y).toEqual(500)
  expect(polyline.points[0].pointType).toEqual('vertex')
})

test('2d polylines drag vertices, midpoints and edges', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // draw a polyline
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[10, 10], [100, 100], [200, 100], [100, 0]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */

  // drag a vertex
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 5)
  label2dHandler.onMouseMove(new Vector2D(300, 100), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(300, 100), 0, 5)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, 0, 1)
  label2dHandler.onMouseDown(new Vector2D(10, 10), 0, 1)
  label2dHandler.onMouseMove(new Vector2D(50, 50), canvasSize, 0, 1)
  label2dHandler.onMouseUp(new Vector2D(50, 50), 0, 1)
  let state = Session.getState()
  let polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points[2].x).toEqual(300)
  expect(polyline.points[2].y).toEqual(100)
  expect(polyline.points[0].x).toEqual(50)
  expect(polyline.points[0].y).toEqual(50)
  expect(polyline.points[3].x).toEqual(100)
  expect(polyline.points[3].y).toEqual(0)
  /**
   * polyline 1: (50, 50) (100, 100) (300, 100) (100, 0)
   */

  // drag midpoints
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 4)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 4)
  label2dHandler.onMouseMove(new Vector2D(200, 150), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(200, 150), 0, 5)
  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points[2].x).toEqual(200)
  expect(polyline.points[2].y).toEqual(150)
  expect(polyline.points[2].pointType).toEqual('vertex')
  expect(polyline.points.length).toEqual(5)
  /**
   * polyline 1: (50, 50) (100, 100) (200, 150) (300, 100) (100, 0)
   */

  // drag edges
  label2dHandler.onMouseMove(new Vector2D(70, 70), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(70, 70), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(170, 170), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(170, 170), 0, 0)
  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points[0].x).toEqual(150)
  expect(polyline.points[0].y).toEqual(150)
  expect(polyline.points[0].pointType).toEqual('vertex')
  expect(polyline.points.length).toEqual(5)
  /**
   * polyline 1: (150, 150) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test('2d polylines delete vertex and draw bezier curve', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // draw a polyline and delete vertex when drawing
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  drawPolygon(label2dHandler, canvasSize,
    [[250, 100], [300, 0], [350, 100], [320, 130], [300, 150]])
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  let polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(5)
  expect(polyline.points[0].x).toEqual(250)
  expect(polyline.points[0].y).toEqual(100)
  expect(polyline.points[0].pointType).toEqual('vertex')
  expect(polyline.points[1].x).toEqual(300)
  expect(polyline.points[1].y).toEqual(0)
  expect(polyline.points[1].pointType).toEqual('vertex')
  expect(polyline.points[2].x).toEqual(350)
  expect(polyline.points[2].y).toEqual(100)
  expect(polyline.points[2].pointType).toEqual('vertex')
  expect(polyline.points[3].x).toEqual(320)
  expect(polyline.points[3].y).toEqual(130)
  expect(polyline.points[3].pointType).toEqual('vertex')
  expect(polyline.points[4].x).toEqual(300)
  expect(polyline.points[4].y).toEqual(150)
  expect(polyline.points[4].pointType).toEqual('vertex')

  // delete vertex when closed
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(325, 50), canvasSize, 0, 4)
  label2dHandler.onMouseDown(new Vector2D(325, 50), 0, 4)
  label2dHandler.onMouseUp(new Vector2D(325, 50), 0, 4)
  label2dHandler.onMouseMove(new Vector2D(300, 150), canvasSize, 0, 9)
  label2dHandler.onMouseDown(new Vector2D(300, 150), 0, 9)
  label2dHandler.onMouseUp(new Vector2D(300, 150), 0, 9)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  /**
   * polyline: (250, 100) (300, 0) (350, 100) (320, 130)
   */

  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(4)
  expect(polyline.points[3].x).toEqual(320)
  expect(polyline.points[3].y).toEqual(130)
  expect(polyline.points[3].pointType).toEqual('vertex')
  expect(polyline.points[0].x).toEqual(250)
  expect(polyline.points[0].y).toEqual(100)
  expect(polyline.points[0].pointType).toEqual('vertex')

  // draw bezier curve
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'c' }))
  label2dHandler.onMouseMove(new Vector2D(335, 115), canvasSize, 0, 6)
  label2dHandler.onMouseDown(new Vector2D(335, 125), 0, 6)
  label2dHandler.onMouseUp(new Vector2D(335, 115), 0, 0)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'c' }))
  /**
   * polyline: (250, 100) (300, 0) (350, 100)
   *          [ (340, 110) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(6)
  expect(polyline.points[3].x).toEqual(340)
  expect(polyline.points[3].y).toEqual(110)
  expect(polyline.points[3].pointType).toEqual('bezier')
  expect(polyline.points[4].x).toEqual(330)
  expect(polyline.points[4].y).toEqual(120)
  expect(polyline.points[4].pointType).toEqual('bezier')

  // drag bezier curve control points
  label2dHandler.onMouseMove(new Vector2D(340, 110), canvasSize, 0, 6)
  label2dHandler.onMouseDown(new Vector2D(340, 110), 0, 6)
  label2dHandler.onMouseMove(new Vector2D(340, 90), canvasSize, 0, 6)
  label2dHandler.onMouseUp(new Vector2D(340, 90), 0, 6)
  /**
   * polyline: (250, 100) (300, 0) (350, 100)
   *          [ (340, 90) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(6)
  expect(polyline.points[2].x).toEqual(350)
  expect(polyline.points[2].y).toEqual(100)
  expect(polyline.points[2].pointType).toEqual('vertex')
  expect(polyline.points[3].x).toEqual(340)
  expect(polyline.points[3].y).toEqual(90)
  expect(polyline.points[3].pointType).toEqual('bezier')
  expect(polyline.points[4].x).toEqual(330)
  expect(polyline.points[4].y).toEqual(120)
  expect(polyline.points[4].pointType).toEqual('bezier')
  expect(polyline.points[5].x).toEqual(320)
  expect(polyline.points[5].y).toEqual(130)
  expect(polyline.points[5].pointType).toEqual('vertex')

  // delete vertex on bezier curve
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(350, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(350, 100), 0, 5)
  label2dHandler.onMouseUp(new Vector2D(350, 100), 0, 5)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  /**
   * polyline: (250, 100) (300, 0) (320, 130)
   */

  state = Session.getState()
  polyline = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polyline.points.length).toEqual(3)
  expect(polyline.points[1].x).toEqual(300)
  expect(polyline.points[1].y).toEqual(0)
  expect(polyline.points[1].pointType).toEqual('vertex')
})

test('Draw human pose', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 4 }))
  const labelIds: IdType[] = []

  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(200, 200), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 200), -1, 0)
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])

  const state = Session.getState()

  const spec = state.task.config.label2DTemplates[
    state.task.config.labelTypes[state.user.select.labelType]
  ]

  const upperLeft = new Vector2D(Infinity, Infinity)
  const bottomRight = new Vector2D(-Infinity, -Infinity)

  for (const point of spec.nodes) {
    upperLeft.x = Math.min(upperLeft.x, point.x)
    upperLeft.y = Math.min(upperLeft.y, point.y)
    bottomRight.x = Math.max(bottomRight.x, point.x)
    bottomRight.y = Math.max(bottomRight.y, point.y)
  }

  const dimensions = new Vector2D(
    bottomRight.x - upperLeft.x, bottomRight.y - upperLeft.y
  )

  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  const labelState = state.task.items[0].labels[labelIds[0]]
  expect(labelState.shapes.length).toEqual(spec.nodes.length)

  const indexedShapes = labelState.shapes.map(
    (id: IdType) => state.task.items[0].shapes[id]
  )

  for (let i = 0; i < indexedShapes.length; i++) {
    const indexed = indexedShapes[i]
    expect(indexed.shapeType).toEqual(ShapeTypeName.NODE_2D)
    const point = indexed as Point2DType
    const templatePoint = spec.nodes[i]
    expect(point.x).toBeCloseTo(
      (templatePoint.x - upperLeft.x) / dimensions.x * 100 + 100
    )
    expect(point.y).toBeCloseTo(
      (templatePoint.y - upperLeft.y) / dimensions.y * 100 + 100
    )
  }
})

test('Draw label2d list to canvas', () => {
  const labelCanvas = createCanvas(200, 200)
  const labelContext = labelCanvas.getContext('2d')
  const controlCanvas = createCanvas(200, 200)
  const controlContext = controlCanvas.getContext('2d')
  const labelIds: IdType[] = []

  const [label2dHandler] = initializeTestingObjects()

  // Draw first box
  const canvasSize = new Size2D(100, 100)
  label2dHandler.onMouseDown(new Vector2D(1, 1), -1, 0)
  for (let i = 1; i <= 10; i += 1) {
    label2dHandler.onMouseMove(new Vector2D(i, i), canvasSize, -1, 0)
    Session.label2dList.redraw(labelContext, controlContext, 1)
  }
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  Session.label2dList.redraw(labelContext, controlContext, 1)

  const state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.push(findNewLabels(
    Session.getState().task.items[0].labels, labelIds)[0])
  const rect = getShape(state, 0, labelIds[0], 0) as RectType
  expect(rect.x1).toEqual(1)
  expect(rect.y1).toEqual(1)
  expect(rect.x2).toEqual(10)
  expect(rect.y2).toEqual(10)
})

test('Change label ordering', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PathPoint2D(0, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(1, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(1, 2, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(0, 2, PointType.CURVE)).toPathPoint()],
    true
  ))
  Session.dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PathPoint2D(3, 4, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(4, 4, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(4, 5, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(3, 5, PointType.CURVE)).toPathPoint()],
    false
  ))
  Session.dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PathPoint2D(10, 11, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(11, 11, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(11, 12, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(10, 12, PointType.CURVE)).toPathPoint()],
    true
  ))

  let state = Session.getState()
  const labelIds = Object.keys(state.task.items[0].labels)
  expect(labelIds.length).toEqual(3)
  expect(state.task.items[0].labels[labelIds[0]].order).toEqual(0)
  expect(state.task.items[0].labels[labelIds[1]].order).toEqual(1)
  expect(state.task.items[0].labels[labelIds[2]].order).toEqual(2)

  const downEvent = new KeyboardEvent('keydown', { key: Key.ARROW_DOWN })
  const upEvent = new KeyboardEvent('keydown', { key: Key.ARROW_UP })
  const bEvent = new KeyboardEvent('keydown', { key: Key.B_LOW })
  const fEvent = new KeyboardEvent('keydown', { key: Key.F_LOW })

  // Move last label back
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelIds[2]))

  label2dHandler.onKeyDown(downEvent)
  label2dHandler.onKeyUp(downEvent)

  state = Session.getState()
  expect(state.task.items[0].labels[labelIds[0]].order).toEqual(0)
  expect(state.task.items[0].labels[labelIds[1]].order).toEqual(2)
  expect(state.task.items[0].labels[labelIds[2]].order).toEqual(1)

  // Move first label forward
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelIds[0]))
  label2dHandler.onKeyDown(upEvent)
  label2dHandler.onKeyUp(upEvent)

  state = Session.getState()
  expect(state.task.items[0].labels[labelIds[0]].order).toEqual(1)
  expect(state.task.items[0].labels[labelIds[1]].order).toEqual(2)
  expect(state.task.items[0].labels[labelIds[2]].order).toEqual(0)

  // Move label in front to back
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  label2dHandler.onKeyDown(bEvent)
  label2dHandler.onKeyUp(bEvent)

  state = Session.getState()
  expect(state.task.items[0].labels[labelIds[0]].order).toEqual(2)
  expect(state.task.items[0].labels[labelIds[1]].order).toEqual(0)
  expect(state.task.items[0].labels[labelIds[2]].order).toEqual(1)

  // Move label in back to front
  Session.dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  label2dHandler.onKeyDown(fEvent)
  label2dHandler.onKeyUp(fEvent)

  state = Session.getState()
  expect(state.task.items[0].labels[labelIds[0]].order).toEqual(1)
  expect(state.task.items[0].labels[labelIds[1]].order).toEqual(2)
  expect(state.task.items[0].labels[labelIds[2]].order).toEqual(0)
})
