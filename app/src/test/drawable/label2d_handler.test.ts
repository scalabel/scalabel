import { createCanvas } from 'canvas'
import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2DHandler } from '../../js/drawable/2d/label2d_handler'
import { getShape } from '../../js/functional/state_util'
import { makeImageViewerConfig } from '../../js/functional/states'
import { PolygonType, RectType } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { testJson } from '../test_image_objects'

/**
 * Initialize Session, label 3d list, label 3d handler
 */
function initializeTestingObjects (): [Label2DHandler, number] {
  Session.devMode = false
  initStore(testJson)
  Session.dispatch(action.addViewerConfig(1, makeImageViewerConfig()))
  const viewerId = 1

  const label2dHandler = new Label2DHandler()
  Session.subscribe(() => {
    const state = Session.getState()
    Session.label2dList.updateState(state)
    label2dHandler.updateState(state)
  })

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  return [label2dHandler, viewerId]
}

test('Draw 2d boxes to label2d list', () => {
  const [label2dHandler] = initializeTestingObjects()

  // Draw first box
  const canvasSize = new Size2D(100, 100)
  label2dHandler.onMouseMove(new Vector2D(1, 1), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(1, 1), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let rect = getShape(state, 0, 0, 0) as RectType
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
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  rect = getShape(state, 0, 1, 0) as RectType
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
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, 2, 0) as RectType
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
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x1).toEqual(16)
  expect(rect.y1).toEqual(17)

  // flip top left and bottom right corner
  label2dHandler.onMouseMove(new Vector2D(16, 17), canvasSize, 1, 1)
  label2dHandler.onMouseDown(new Vector2D(16, 17), 1, 1)
  label2dHandler.onMouseMove(new Vector2D(42, 43), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(40, 41), -1, 0)
  state = Session.getState()
  rect = getShape(state, 0, 1, 0) as RectType
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
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x1).toEqual(34)
  expect(rect.y1).toEqual(30)
  expect(rect.x2).toEqual(46)
  expect(rect.y2).toEqual(44)

  // delete label
  Session.dispatch(action.deleteLabel(0, 1))
  expect(Session.label2dList.labelList.length).toEqual(2)
  expect(Session.label2dList.labelList[0].index).toEqual(0)
  expect(Session.label2dList.labelList[0].labelId).toEqual(0)
  expect(Session.label2dList.labelList[1].index).toEqual(1)
  expect(Session.label2dList.labelList[1].labelId).toEqual(2)
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
  expect(Session.label2dList.labelList.length).toEqual(2)
})

test('2d polygons highlighted and selected', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(120, 120), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(210, 210), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(210, 210), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(210, 210), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(310, 260), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(310, 260), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(310, 260), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(410, 210), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(410, 210), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(410, 210), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(210, 110), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(210, 110), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(210, 110), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(120, 120), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(120, 120), -1, 1)
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  let selected = Session.label2dList.selectedLabels
  expect(selected[0].labelId).toEqual(0)

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
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // change highlighted label
  label2dHandler.onMouseMove(new Vector2D(130, 130), canvasSize, 0, 0)
  let highlighted = label2dHandler.highlightedLabel
  selected = Session.label2dList.selectedLabels
  expect(Session.label2dList.labelList.length).toEqual(2)
  expect(Session.label2dList.labelList[1].labelId).toEqual(1)
  if (!highlighted) {
    throw new Error('no highlightedLabel')
  } else {
    expect(highlighted.labelId).toEqual(0)
  }
  expect(selected[0].labelId).toEqual(1)

  // change selected label
  label2dHandler.onMouseDown(new Vector2D(130, 130), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(140, 140), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(140, 140), 0, 0)
  highlighted = label2dHandler.highlightedLabel
  selected = Session.label2dList.selectedLabels
  if (highlighted) {
    expect(highlighted.labelId).toEqual(0)
  }
  expect(selected[0].labelId).toEqual(0)
})

test('validation check for polygon2d', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw a valid polygon
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(120, 120), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(210, 210), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(210, 210), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(210, 210), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(310, 260), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(310, 260), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(310, 260), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(410, 210), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(410, 210), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(410, 210), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(210, 110), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(210, 110), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(210, 110), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(120, 120), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(120, 120), -1, 1)
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */

  // draw one invalid polygon
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(400, 300), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(400, 300), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(400, 300), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 200), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 200), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 200), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 0), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 0), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 1)
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
  const polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points[2].x).toEqual(310)
  expect(polygon.points[2].y).toEqual(260)
  expect(polygon.points[2].type).toEqual('vertex')
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */

  // draw a too small polygon
  label2dHandler.onMouseMove(new Vector2D(0, 0), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(0, 0), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(0, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(1, 0), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(1, 0), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(1, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(0, 1), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(0, 1), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(0, 1), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(0, 0), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(0, 0), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(0, 0), -1, 1)
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

  // draw a polygon
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
  label2dHandler.onMouseMove(new Vector2D(100, 0), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(100, 0), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(100, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */

  // drag a vertex
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 5)
  label2dHandler.onMouseMove(new Vector2D(300, 100), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(300, 100), 0, 5)
  let state = Session.getState()
  let polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points[2].x).toEqual(300)
  expect(polygon.points[2].y).toEqual(100)
  expect(polygon.points[2].type).toEqual('vertex')
  /**
   * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
   */

  // drag midpoints
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, 0, 4)
  label2dHandler.onMouseDown(new Vector2D(200, 100), 0, 4)
  label2dHandler.onMouseMove(new Vector2D(200, 150), canvasSize, 0, 5)
  label2dHandler.onMouseUp(new Vector2D(200, 150), 0, 5)
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
  label2dHandler.onMouseMove(new Vector2D(20, 20), canvasSize, 0, 0)
  label2dHandler.onMouseDown(new Vector2D(20, 20), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(120, 120), canvasSize, 0, 0)
  label2dHandler.onMouseUp(new Vector2D(120, 120), 0, 0)
  state = Session.getState()
  polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(110)
  expect(polygon.points[0].y).toEqual(110)
  expect(polygon.points[0].type).toEqual('vertex')
  expect(polygon.points.length).toEqual(5)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test('2d polygons delete vertex and draw bezier curve', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw a polygon and delete vertex when drawing
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(250, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(250, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(250, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 0), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 0), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 0), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(350, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(350, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(350, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 200), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 200), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 200), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(320, 130), canvasSize, -1, 0)
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  label2dHandler.onMouseDown(new Vector2D(320, 130), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(320, 130), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 150), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 150), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 150), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(250, 100), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(250, 100), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(250, 100), -1, 1)
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
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(275, 125), canvasSize, 0, 10)
  label2dHandler.onMouseDown(new Vector2D(275, 125), 0, 10)
  label2dHandler.onMouseUp(new Vector2D(2750, 1250), 0, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 150), canvasSize, 0, 9)
  label2dHandler.onMouseDown(new Vector2D(300, 150), 0, 9)
  label2dHandler.onMouseUp(new Vector2D(300, 150), 0, 9)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
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
  polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points.length).toEqual(6)
  expect(polygon.points[3].x).toEqual(340)
  expect(polygon.points[3].y).toEqual(110)
  expect(polygon.points[3].type).toEqual('bezier')
  expect(polygon.points[4].x).toEqual(330)
  expect(polygon.points[4].y).toEqual(120)
  expect(polygon.points[4].type).toEqual('bezier')

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
  label2dHandler.onKeyDown(new KeyboardEvent('keydown', { key: 'd' }))
  label2dHandler.onMouseMove(new Vector2D(350, 100), canvasSize, 0, 5)
  label2dHandler.onMouseDown(new Vector2D(350, 100), 0, 5)
  label2dHandler.onMouseUp(new Vector2D(350, 100), 0, 5)
  label2dHandler.onKeyUp(new KeyboardEvent('keyup', { key: 'd' }))
  /**
   * polygon: (250, 100) (300, 0) (320, 130)
   */

  state = Session.getState()
  polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points.length).toEqual(3)
})

test('2d polygons multi-select and multi-label moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

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
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // draw third polygon
  label2dHandler.onMouseMove(new Vector2D(250, 250), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(250, 250), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(250, 250), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 250), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 250), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(350, 350), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(350, 350), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(350, 350), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(250, 250), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(250, 250), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(250, 250), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

   // select label 1
  label2dHandler.onMouseMove(new Vector2D(600, 600), canvasSize, 1, 0)
  label2dHandler.onMouseDown(new Vector2D(600, 600), 1, 0)
  label2dHandler.onMouseUp(new Vector2D(600, 600), 1, 0)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(1)
  expect(state.user.select.labels[0][0]).toEqual(1)
  expect(Session.label2dList.selectedLabels.length).toEqual(1)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(1)

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
  let polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(110)
  expect(polygon.points[0].y).toEqual(110)
  polygon = getShape(state, 0, 1, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(600)
  expect(polygon.points[0].y).toEqual(600)
  polygon = getShape(state, 0, 2, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(350)
  expect(polygon.points[0].y).toEqual(350)
})

test('2d polygons linking labels and moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  label2dHandler.onMouseDown(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(10, 10), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(100, 100), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(100, 100), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(200, 100), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(200, 100), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(200, 100), -1, 1)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

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
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // draw third polygon
  label2dHandler.onMouseMove(new Vector2D(250, 250), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(250, 250), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(250, 250), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(300, 250), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(300, 250), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(300, 250), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(350, 350), canvasSize, -1, 0)
  label2dHandler.onMouseDown(new Vector2D(350, 350), -1, 0)
  label2dHandler.onMouseUp(new Vector2D(350, 350), -1, 0)
  label2dHandler.onMouseMove(new Vector2D(250, 250), canvasSize, -1, 1)
  label2dHandler.onMouseDown(new Vector2D(250, 250), -1, 1)
  label2dHandler.onMouseUp(new Vector2D(250, 250), -1, 1)
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
  expect(state.user.select.labels[0][0]).toEqual(2)
  expect(state.user.select.labels[0][1]).toEqual(0)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(2)
  expect(Session.label2dList.selectedLabels[1].labelId).toEqual(0)

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
  let polygon = getShape(state, 0, 0, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(110)
  expect(polygon.points[0].y).toEqual(110)
  polygon = getShape(state, 0, 1, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(600)
  expect(polygon.points[0].y).toEqual(600)
  polygon = getShape(state, 0, 2, 0) as PolygonType
  expect(polygon.points[0].x).toEqual(250)
  expect(polygon.points[0].y).toEqual(250)
})

test('Draw label2d list to canvas', () => {
  const labelCanvas = createCanvas(200, 200)
  const labelContext = labelCanvas.getContext('2d')
  const controlCanvas = createCanvas(200, 200)
  const controlContext = controlCanvas.getContext('2d')

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
  const rect = getShape(state, 0, 0, 0) as RectType
  expect(rect.x1).toEqual(1)
  expect(rect.y1).toEqual(1)
  expect(rect.x2).toEqual(10)
  expect(rect.y2).toEqual(10)
})
