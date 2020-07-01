import { createCanvas } from 'canvas'
import _ from 'lodash'
import * as action from '../../js/action/common'
import { addPolygon2dLabel } from '../../js/action/polygon2d'
import { selectLabel } from '../../js/action/select'
import Session, { dispatch, getState } from '../../js/common/session'
import { Key, ShapeTypeName } from '../../js/common/types'
import { Label2DHandler } from '../../js/drawable/2d/label2d_handler'
import { PolyPathPoint2D, PointType } from '../../js/drawable/2d/poly_path_point2d'
import { getNumLabels, getShape } from '../../js/functional/state_util'
import { makeImageViewerConfig } from '../../js/functional/states'
import { IdType, Point2DType, PolygonType, RectType, SimpleRect } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { setupTestStore } from '../components/util'
import { drawBox2D as drawBox2D, drawPolygon, drawPolygonByDragging, keyClick, keyDown, keyUp,
  mouseClick, mouseDown, mouseMove, mouseMoveClick,
  mouseUp, moveBox2D as moveBox2D, resizeBox2D as resizeBox2D } from '../drawable/label2d_handler_util'
import { findNewLabels, findNewLabelsFromState } from '../server/util/util'
import { testJson } from '../test_states/test_image_objects'
import { checkBox2D, checkPolygon } from '../util/shape'

/**
 * Initialize Session, label 2d list, label 2d handler
 */
function initializeTestingObjects (): [Label2DHandler, number] {
  setupTestStore(testJson)

  dispatch(action.addViewerConfig(1, makeImageViewerConfig(0)))
  const viewerId = 1

  const label2dHandler = new Label2DHandler(Session.label2dList)
  Session.subscribe(() => {
    const state = getState()
    Session.label2dList.updateState(state)
    label2dHandler.updateState(state)
  })

  dispatch(action.loadItem(0, -1))
  dispatch(action.goToItem(0))

  return [label2dHandler, viewerId]
}

describe('Draw 2D boxes to label2d list', () => {
  // Samples box2D coords to use for tests
  const boxCoords: SimpleRect[] = [
    { x1: 1, y1: 1, x2: 10, y2: 10 },
    { x1: 19, y1: 20, x2: 30, y2: 29 },
    { x1: 4, y1: 5, x2: 23, y2: 24 }
  ]

  test('Add and delete boxes', () => {
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(100, 100)
    const labelIds: IdType[] = []

    // Draw and check each box
    for (const coords of boxCoords) {
      const labelId = drawBox2D(label2dHandler, canvasSize, coords)
      checkBox2D(labelId, coords)
      labelIds.push(labelId)
    }

    // Delete label
    dispatch(action.deleteLabel(0, labelIds[1]))
    const labelList = Session.label2dList.labelList
    expect(labelList.length).toEqual(2)
    expect(labelList[0].index).toEqual(0)
    expect(labelList[0].labelId).toEqual(labelIds[0])
    expect(labelList[1].index).toEqual(1)
    expect(labelList[1].labelId).toEqual(labelIds[2])
  })

  test('Add boxes with interrupting actions', () => {
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(100, 100)
    const interrupt = true

    // Draw and check each box
    for (const coords of boxCoords) {
      const labelId = drawBox2D(label2dHandler, canvasSize, coords, interrupt)
      checkBox2D(labelId, coords)
    }
  })

  test('Resize and move boxes', () => {
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(100, 100)
    const labelIds: IdType[] = []

    // Draw each box
    for (const coords of boxCoords) {
      const labelId = drawBox2D(label2dHandler, canvasSize, coords)
      checkBox2D(labelId, coords)
      labelIds.push(labelId)
    }

    // Resize the second box
    const boxIndex = 1
    const boxId = labelIds[boxIndex]
    const originalCoords = boxCoords[boxIndex]
    let moveCoords = {
      x1: originalCoords.x1, y1: originalCoords.y1, x2: 16, y2: 17}
    resizeBox2D(label2dHandler, canvasSize, moveCoords, boxIndex)
    checkBox2D(boxId, { x1: 16, y1: 17, x2: 30, y2: 29 })

    // Flip top left and bottom right corners
    moveCoords = {
      x1: moveCoords.x2, y1: moveCoords.y2, x2: 42, y2: 43
    }
    resizeBox2D(label2dHandler, canvasSize, moveCoords, boxIndex)
    checkBox2D(boxId, { x1: 30, y1: 29, x2: 42, y2: 43 })

    // Move the entire box +4x and -1y
    moveCoords = {
      x1: 32, y1: 31, x2: 36, y2: 32
    }
    moveBox2D(label2dHandler, canvasSize, moveCoords, boxIndex)
    checkBox2D(boxId, { x1: 34, y1: 30, x2: 46, y2: 44 })
  })
})

describe('Draw 2d polygons to label2d list', () => {
  // Samples polygon vertices to use for tests
  const vertices: number[][][] = [
    [[10, 10], [100, 100], [200, 100], [100, 0]],
    [[500, 500], [600, 400], [700, 700]]
  ]

  test('Draw polygon with a mix of clicking and dragging', () => {
    const itemIndex = 0
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(1000, 1000)
    dispatch(action.changeSelect({ labelType: 1 }))

    // Draw the first points by clicking
    mouseMoveClick(label2dHandler, 10, 10, canvasSize, -1, 0)
    mouseMoveClick(label2dHandler, 100, 100, canvasSize, -1, 0)
    mouseMoveClick(label2dHandler, 200, 100, canvasSize, -1, 0)

    // Check that polygon isn't added to state until it's finished
    expect(getNumLabels(getState(), itemIndex)).toEqual(0)

    // Drag when drawing the last point
    mouseMove(label2dHandler, 200, 10, canvasSize, -1, 0)
    mouseDown(label2dHandler, 200, 10, -1, 0)
    mouseMove(label2dHandler, 100, 0, canvasSize, -1, 0)
    mouseUp(label2dHandler, 100, 0, -1, 0)
    mouseMoveClick(label2dHandler, 10, 10, canvasSize, -1, 1)

    const labelId = findNewLabelsFromState(getState(), itemIndex, [])[0]
    checkPolygon(labelId, vertices[0])
  })

  test('Draw multiple polygons', () => {
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(1000, 1000)
    dispatch(action.changeSelect({ labelType: 1 }))

    const labelIds: IdType[] = []

    labelIds.push(drawPolygon(label2dHandler, canvasSize, vertices[0]))
    checkPolygon(labelIds[0], vertices[0])

    labelIds.push(drawPolygonByDragging(
      label2dHandler, canvasSize, vertices[1]))
    checkPolygon(labelIds[1], vertices[1])

    expect(Session.label2dList.labelList.length).toEqual(2)
  })

  test('Draw polygons with interrupting actions', () => {
    const [label2dHandler] = initializeTestingObjects()
    const canvasSize = new Size2D(1000, 1000)
    const interrupt = true
    dispatch(action.changeSelect({ labelType: 1 }))

    const labelIds: IdType[] = []

    labelIds.push(drawPolygon(
      label2dHandler, canvasSize, vertices[0], interrupt))
    checkPolygon(labelIds[0], vertices[0])

    labelIds.push(drawPolygonByDragging(
      label2dHandler, canvasSize, vertices[1], interrupt))
    checkPolygon(labelIds[1], vertices[1])

    expect(Session.label2dList.labelList.length).toEqual(2)
  })
})

test('2d polygons highlighted and selected', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))

  const labelIds: IdType[] = []

  // Draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  let selected = Session.label2dList.selectedLabels
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  expect(selected[0].labelId).toEqual(labelIds[0])

  // Draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  // Change highlighted label
  mouseMove(label2dHandler, 130, 130, canvasSize, 0, 0)
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

  // Change selected label
  mouseDown(label2dHandler, 130, 130, 0, 0)
  mouseMove(label2dHandler, 140, 140, canvasSize, 0, 0)
  mouseUp(label2dHandler, 140, 140, 0, 0)
  highlighted = label2dHandler.highlightedLabel
  selected = Session.label2dList.selectedLabels
  if (highlighted) {
    expect(highlighted.labelId).toEqual(labelIds[0])
  }
  expect(selected[0].labelId).toEqual(labelIds[0])
})

test('validation check for polygon2d', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw a valid polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[120, 120], [210, 210], [310, 260], [410, 210], [210, 110]])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  // Draw one invalid polygon
  drawPolygon(label2dHandler, canvasSize,
    [[200, 100], [400, 300], [300, 200], [300, 0]], false, false)
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (200, 100) (400, 300) (300, 200) (300, 0) invalid
   */

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  // Drag the polygon to an invalid shape
  mouseMove(label2dHandler, 310, 260, canvasSize, 0, 5)
  mouseDown(label2dHandler, 310, 260, 0, 5)
  mouseMove(label2dHandler, 310, 0, canvasSize, 0, 5)
  mouseUp(label2dHandler, 310, 0, 0, 5)
  /**
   * polygon 1: (120, 120) (210, 210) (310, 0) (410, 210) (210, 110)
   * polygon 1 is an invalid shape
   */

  state = getState()
  const polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[2].x).toEqual(310)
  expect(polygon.points[2].y).toEqual(260)
  expect(polygon.points[2].pointType).toEqual('vertex')
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, [])[0])

  // Draw a too small polygon
  drawPolygon(label2dHandler, canvasSize, [[0, 0], [1, 0], [0, 1]],
              false, false)
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (0, 0) (1, 0) (0, 1) too small, invalid
   */
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)
})

test('2d polygons drag vertices, midpoints and edges', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw a polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[10, 10], [100, 100], [200, 100], [100, 0]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  // Drag a vertex
  mouseMove(label2dHandler, 200, 100, canvasSize, 0, 5)
  mouseDown(label2dHandler, 200, 100, 0, 5)
  mouseMove(label2dHandler, 300, 100, canvasSize, 0, 5)
  mouseUp(label2dHandler, 300, 100, 0, 5)
  let state = getState()
  let points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points[2]).toMatchObject({ x: 300, y: 100, pointType: 'vertex' })

  /**
   * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
   */

  // Drag midpoints
  mouseMove(label2dHandler, 200, 100, canvasSize, 0, 4)
  mouseDown(label2dHandler, 200, 100, 0, 4)
  mouseMove(label2dHandler, 200, 150, canvasSize, 0, 5)
  mouseUp(label2dHandler, 200, 150, 0, 5)
  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[2]).toMatchObject({ x: 200, y: 150, pointType: 'vertex' })

  /**
   * polygon 1: (10, 10) (100, 100) (200, 150) (300, 100) (100, 0)
   */

  // Drag edges
  mouseMove(label2dHandler, 20, 20, canvasSize, 0, 0)
  mouseDown(label2dHandler, 20, 20, 0, 0)
  mouseMove(label2dHandler, 120, 120, canvasSize, 0, 0)
  mouseUp(label2dHandler, 120, 120, 0, 0)
  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[0]).toMatchObject({ x: 110, y: 110, pointType: 'vertex' })

  /**
   * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test('2d polygons delete vertex and draw bezier curve', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw a polygon and delete vertex when drawing
  const canvasSize = new Size2D(1000, 1000)
  mouseMoveClick(label2dHandler, 200, 100, canvasSize, -1, 0)
  keyClick(label2dHandler, 'd')
  mouseMoveClick(label2dHandler, 250, 100, canvasSize, -1, 0)
  mouseMoveClick(label2dHandler, 300, 0, canvasSize, -1, 0)
  mouseMoveClick(label2dHandler, 350, 100, canvasSize, -1, 0)
  mouseMoveClick(label2dHandler, 300, 200, canvasSize, -1, 0)
  mouseMove(label2dHandler, 320, 130, canvasSize, -1, 0)
  keyClick(label2dHandler, 'd')
  mouseClick(label2dHandler, 320, 130, -1, 0)
  mouseMoveClick(label2dHandler, 300, 150, canvasSize, -1, 0)
  mouseMoveClick(label2dHandler, 250, 100, canvasSize, -1, 1)

  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  let points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[0]).toMatchObject({ x: 250, y: 100, pointType: 'vertex' })
  expect(points[1]).toMatchObject({ x: 300, y: 0, pointType: 'vertex' })
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: 'vertex' })
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: 'vertex' })
  expect(points[4]).toMatchObject({ x: 300, y: 150, pointType: 'vertex' })

  // Delete vertex when closed
  keyDown(label2dHandler, 'd')
  mouseMoveClick(label2dHandler, 275, 125, canvasSize, 0, 10)
  mouseMoveClick(label2dHandler, 300, 150, canvasSize, 0, 9)
  keyUp(label2dHandler, 'd')
  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(4)
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: 'vertex' })

  // Draw bezier curve
  keyDown(label2dHandler, 'c')
  mouseMoveClick(label2dHandler, 335, 115, canvasSize, 0, 6)
  keyUp(label2dHandler, 'c')
  /**
   * polygon: (250, 100) (300, 0) (350, 100)
   *          [ (340, 110) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(6)
  expect(points[3]).toMatchObject({ x: 340, y: 110, pointType: 'bezier' })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: 'bezier' })

  // Drag bezier curve control points
  mouseMove(label2dHandler, 340, 110, canvasSize, 0, 6)
  mouseDown(label2dHandler, 340, 110, 0, 6)
  mouseMove(label2dHandler, 340, 90, canvasSize, 0, 6)
  mouseUp(label2dHandler, 340, 90, 0, 6)
  /**
   * polygon: (250, 100) (300, 0) (350, 100)
   *          [ (340, 90) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(6)
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: 'vertex' })
  expect(points[3]).toMatchObject({ x: 340, y: 90, pointType: 'bezier' })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: 'bezier' })

  // Delete vertex on bezier curve
  keyDown(label2dHandler, 'd')
  mouseMoveClick(label2dHandler, 350, 100, canvasSize, 0, 5)
  keyUp(label2dHandler, 'd')
  /**
   * polygon: (250, 100) (300, 0) (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(3)
})

test('2d polygon select and moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  // Select label 1
  mouseMoveClick(label2dHandler, 50, 50, canvasSize, 0, 0)

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(1)
  expect(state.user.select.labels[0][0]).toEqual(labelIds[0])
  const label2dlist = Session.label2dList
  expect(label2dlist.selectedLabels.length).toEqual(1)
  expect(label2dlist.selectedLabels[0].labelId).toEqual(labelIds[0])

  mouseMove(label2dHandler, 20, 20, canvasSize, 0, 0)
  mouseDown(label2dHandler, 20, 20, 0, 0)
  mouseMove(label2dHandler, 60, 60, canvasSize, 0, 0)
  mouseMove(label2dHandler, 120, 120, canvasSize, 0, 0)

  mouseUp(label2dHandler, 120, 120, 0, 0)

  state = getState()
  const polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0].x).toEqual(110)
  expect(polygon.points[0].y).toEqual(110)
})

test('2d polygons multi-select and multi-label moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  // Draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  // Draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select label 1
  mouseMoveClick(label2dHandler, 600, 600, canvasSize, 1, 0)

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(1)
  expect(state.user.select.labels[0][0]).toEqual(labelIds[1])
  expect(Session.label2dList.selectedLabels.length).toEqual(1)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(labelIds[1])

  // Select label 1, 2, 3
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 300, 250, canvasSize, 2, 2)
  mouseMoveClick(label2dHandler, 50, 50, canvasSize, 0, 0)
  keyUp(label2dHandler, 'Meta')

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // Unselect label 3
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 300, 250, canvasSize, 2, 2)
  keyUp(label2dHandler, 'Meta')

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)

  // Select three labels
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 300, 250, canvasSize, 2, 2)
  keyUp(label2dHandler, 'Meta')

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // Move
  mouseMove(label2dHandler, 20, 20, canvasSize, 0, 0)
  mouseDown(label2dHandler, 20, 20, 0, 0)
  mouseMove(label2dHandler, 60, 60, canvasSize, 0, 0)
  mouseMove(label2dHandler, 120, 120, canvasSize, 0, 0)
  mouseUp(label2dHandler, 120, 120, 0, 0)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (350, 350) (400, 350) (450, 450)
   */

  state = getState()
  let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 110, y: 110 })
  polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 600, y: 600 })
  polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 350, y: 350 })
})

test('2d polygons linking labels and moving', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds: IdType[] = []

  // Draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])

  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

  // Draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // Draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select label 2 and 0
  mouseMoveClick(label2dHandler, 300, 300, canvasSize, 2, 0)
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 100, 100, canvasSize, 0, 2)
  keyUp(label2dHandler, 'Meta')
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(state.user.select.labels[0][0]).toEqual(labelIds[2])
  expect(state.user.select.labels[0][1]).toEqual(labelIds[0])
  expect(Session.label2dList.selectedLabels.length).toEqual(2)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(labelIds[2])
  expect(Session.label2dList.selectedLabels[1].labelId).toEqual(labelIds[0])

  // Select label 1 and 2
  mouseMoveClick(label2dHandler, 600, 600, canvasSize, 1, 0)
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 50, 50, canvasSize, 0, 0)
  keyUp(label2dHandler, 'Meta')

  // Link label 1 and 2
  keyClick(label2dHandler, 'l')
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(4)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)
  expect(Session.label2dList.labelList[0].color).toEqual(
    Session.label2dList.labelList[1].color
  )

  // Reselect label 1 and 2
  mouseMoveClick(label2dHandler, 300, 250, canvasSize, 2, 2)
  mouseMoveClick(label2dHandler, 50, 50, canvasSize, 0, 0)

  state = getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)

  // Moving group 1
  mouseMove(label2dHandler, 20, 20, canvasSize, 0, 0)
  mouseDown(label2dHandler, 20, 20, 0, 0)
  mouseMove(label2dHandler, 60, 60, canvasSize, 0, 0)
  mouseMove(label2dHandler, 120, 120, canvasSize, 0, 0)
  mouseUp(label2dHandler, 120, 120, 0, 0)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = getState()
  let polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 110, y: 110 })
  polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 600, y: 600 })
  polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 250, y: 250 })

  // Reshape for one label in group
  mouseMove(label2dHandler, 110, 110, canvasSize, 0, 1)
  mouseDown(label2dHandler, 110, 110, 0, 1)
  mouseMove(label2dHandler, 100, 100, canvasSize, 0, 1)
  mouseUp(label2dHandler, 100, 100, 0, 1)
  /**
   * polygon 1: (100, 100) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = getState()
  polygon = getShape(state, 0, labelIds[0], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 100, y: 100 })
  polygon = getShape(state, 0, labelIds[1], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 600, y: 600 })
  polygon = getShape(state, 0, labelIds[2], 0) as PolygonType
  expect(polygon.points[0]).toMatchObject({ x: 250, y: 250 })
})

test('2d polygons unlinking', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 1 }))

  // Draw first polygon
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

  // Draw second polygon
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // Draw third polygon
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select polygon 1 and 3
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 100, 100, canvasSize, 0, 2)
  keyUp(label2dHandler, 'Meta')

  // Link polygon 1 and 3
  keyClick(label2dHandler, 'l')

  state = getState()
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

  // Select polygon 1, 2, 3
  mouseMoveClick(label2dHandler, 550, 550, canvasSize, 1, 0)
  keyDown(label2dHandler, 'Meta')
  mouseMoveClick(label2dHandler, 100, 100, canvasSize, 0, 2)
  keyUp(label2dHandler, 'Meta')

  // Unlink polygon 1 and 3
  keyClick(label2dHandler, 'L')
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)

  // Unselect polygon 1
  keyDown(label2dHandler, 'Meta')
  mouseMove(label2dHandler, 100, 100, canvasSize, 0, 2)
  mouseDown(label2dHandler, 100, 100, 0, 2)
  mouseUp(label2dHandler, 100, 100, 0, 2)
  keyUp(label2dHandler, 'Meta')

  // Link polygon 2 and 3
  keyClick(label2dHandler, 'l')
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: polygon 2, 3
   */

  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(4)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)
  expect(Session.label2dList.labelList[1].color).toEqual(
    Session.label2dList.labelList[2].color)
})

test('2d polyline creating', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // Draw first polyline
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize, [[10, 10], [100, 100], [200, 100]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   */

  // Draw second polyline
  drawPolygon(label2dHandler, canvasSize, [[500, 500], [600, 400], [700, 700]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   * polyline 2: (500, 500) (600, 400) (700, 700)
   */

  // Draw third polyline
  drawPolygon(label2dHandler, canvasSize, [[250, 250], [300, 250], [350, 350]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100)
   * polyline 2: (500, 500) (600, 400) (700, 700)
   * polyline 3: (250, 250) (300, 250) (350, 350)
   */

  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  let points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(3)
  expect(points[0]).toMatchObject({ x: 10, y: 10, pointType: 'vertex' })
  expect(points[1]).toMatchObject({ x: 100, y: 100, pointType: 'vertex' })
  expect(points[2]).toMatchObject({ x: 200, y: 100, pointType: 'vertex' })

  points = (getShape(state, 0, labelIds[1], 0) as PolygonType).points
  expect(points.length).toEqual(3)
  expect(points[0]).toMatchObject({ x: 500, y: 500, pointType: 'vertex' })

  points = (getShape(state, 0, labelIds[1], 0) as PolygonType).points
  expect(points.length).toEqual(3)
  expect(points[0]).toMatchObject({ x: 500, y: 500, pointType: 'vertex' })
})

test('2d polylines drag vertices, midpoints and edges', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // Draw a polyline
  const canvasSize = new Size2D(1000, 1000)
  drawPolygon(label2dHandler, canvasSize,
    [[10, 10], [100, 100], [200, 100], [100, 0]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */

  // Drag a vertex
  mouseMove(label2dHandler, 200, 100, canvasSize, 0, 5)
  mouseDown(label2dHandler, 200, 100, 0, 5)
  mouseMove(label2dHandler, 300, 100, canvasSize, 0, 5)
  mouseUp(label2dHandler, 300, 100, 0, 5)
  mouseMove(label2dHandler, 10, 10, canvasSize, 0, 1)
  mouseDown(label2dHandler, 10, 10, 0, 1)
  mouseMove(label2dHandler, 50, 50, canvasSize, 0, 1)
  mouseUp(label2dHandler, 50, 50, 0, 1)
  let state = getState()
  let points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points[0]).toMatchObject({ x: 50, y: 50 })
  expect(points[2]).toMatchObject({ x: 300, y: 100 })
  expect(points[3]).toMatchObject({ x: 100, y: 0 })

  /**
   * polyline 1: (50, 50) (100, 100) (300, 100) (100, 0)
   */

  // Drag midpoints
  mouseMove(label2dHandler, 200, 100, canvasSize, 0, 4)
  mouseDown(label2dHandler, 200, 100, 0, 4)
  mouseMove(label2dHandler, 200, 150, canvasSize, 0, 5)
  mouseUp(label2dHandler, 200, 150, 0, 5)
  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[2]).toMatchObject({ x: 200, y: 150, pointType: 'vertex' })
  /**
   * polyline 1: (50, 50) (100, 100) (200, 150) (300, 100) (100, 0)
   */

  // Drag edges
  mouseMove(label2dHandler, 70, 70, canvasSize, 0, 0)
  mouseDown(label2dHandler, 70, 70, 0, 0)
  mouseMove(label2dHandler, 170, 170, canvasSize, 0, 0)
  mouseUp(label2dHandler, 170, 170, 0, 0)
  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[0]).toMatchObject({ x: 150, y: 150, pointType: 'vertex' })

  /**
   * polyline 1: (150, 150) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test('2d polylines delete vertex and draw bezier curve', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 3 }))
  const labelIds: IdType[] = []

  // Draw a polyline and delete vertex when drawing
  const canvasSize = new Size2D(1000, 1000)
  mouseMove(label2dHandler, 200, 100, canvasSize, -1, 0)
  mouseUp(label2dHandler, 200, 100, -1, 0)
  mouseDown(label2dHandler, 200, 100, -1, 0)
  keyClick(label2dHandler, 'd')
  drawPolygon(label2dHandler, canvasSize,
    [[250, 100], [300, 0], [350, 100], [320, 130], [300, 150]])
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  /**
   * polyline: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
   */

  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  let points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(5)
  expect(points[0]).toMatchObject({ x: 250, y: 100, pointType: 'vertex' })
  expect(points[1]).toMatchObject({ x: 300, y: 0, pointType: 'vertex' })
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: 'vertex' })
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: 'vertex' })
  expect(points[4]).toMatchObject({ x: 300, y: 150, pointType: 'vertex' })

  // Delete vertex when closed
  keyDown(label2dHandler, 'd')
  mouseMoveClick(label2dHandler, 325, 50, canvasSize, 0, 4)
  mouseMoveClick(label2dHandler, 300, 150, canvasSize, 0, 9)
  keyUp(label2dHandler, 'd')
  /**
   * polyline: (250, 100) (300, 0) (350, 100) (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(4)
  expect(points[0]).toMatchObject({ x: 250, y: 100, pointType: 'vertex' })
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: 'vertex' })

  // Draw bezier curve
  keyDown(label2dHandler, 'c')
  mouseMoveClick(label2dHandler, 335, 115, canvasSize, 0, 6)
  keyUp(label2dHandler, 'c')
  /**
   * polyline: (250, 100) (300, 0) (350, 100)
   *          [ (340, 110) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(6)
  expect(points[3]).toMatchObject({ x: 340, y: 110, pointType: 'bezier' })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: 'bezier' })

  // Drag bezier curve control points
  mouseMove(label2dHandler, 340, 110, canvasSize, 0, 6)
  mouseDown(label2dHandler, 340, 110, 0, 6)
  mouseMove(label2dHandler, 340, 90, canvasSize, 0, 6)
  mouseUp(label2dHandler, 340, 90, 0, 6)
  /**
   * polyline: (250, 100) (300, 0) (350, 100)
   *          [ (340, 90) (330, 120) <bezier curve control points>]
   *          (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(6)
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: 'vertex' })
  expect(points[3]).toMatchObject({ x: 340, y: 90, pointType: 'bezier' })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: 'bezier' })
  expect(points[5]).toMatchObject({ x: 320, y: 130, pointType: 'vertex' })

  // Delete vertex on bezier curve
  keyDown(label2dHandler, 'd')
  mouseMove(label2dHandler, 350, 100, canvasSize, 0, 5)
  mouseDown(label2dHandler, 350, 100, 0, 5)
  mouseUp(label2dHandler, 350, 100, 0, 5)
  keyUp(label2dHandler, 'd')
  /**
   * polyline: (250, 100) (300, 0) (320, 130)
   */

  state = getState()
  points = (getShape(state, 0, labelIds[0], 0) as PolygonType).points
  expect(points.length).toEqual(3)
  expect(points[1]).toMatchObject({ x: 300, y: 0, pointType: 'vertex' })
})

test('Draw human pose', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 4 }))
  const labelIds: IdType[] = []

  const canvasSize = new Size2D(1000, 1000)
  mouseMove(label2dHandler, 100, 100, canvasSize, -1, 0)
  mouseDown(label2dHandler, 100, 100, -1, 0)
  mouseMove(label2dHandler, 200, 200, canvasSize, -1, 0)
  mouseUp(label2dHandler, 200, 200, -1, 0)
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])

  const state = getState()

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
  mouseDown(label2dHandler, 1, 1, -1, 0)
  for (let i = 1; i <= 10; i += 1) {
    mouseMove(label2dHandler, i, i, canvasSize, -1, 0)
    Session.label2dList.redraw(labelContext, controlContext, 1)
  }
  mouseUp(label2dHandler, 10, 10, -1, 0)
  Session.label2dList.redraw(labelContext, controlContext, 1)

  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.push(findNewLabels(
    getState().task.items[0].labels, labelIds)[0])
  const rect = getShape(state, 0, labelIds[0], 0) as RectType
  expect(rect).toMatchObject({ x1: 1, y1: 1, x2: 10, y2: 10 })
})

test('Change label ordering', () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PolyPathPoint2D(0, 1, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(1, 1, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(1, 2, PointType.CURVE)).toPathPoint(),
      (new PolyPathPoint2D(0, 2, PointType.CURVE)).toPathPoint()],
    true
  ))
  dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PolyPathPoint2D(3, 4, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(4, 4, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(4, 5, PointType.CURVE)).toPathPoint(),
      (new PolyPathPoint2D(3, 5, PointType.CURVE)).toPathPoint()],
    false
  ))
  dispatch(addPolygon2dLabel(
    0,
    -1,
    [0],
    [(new PolyPathPoint2D(10, 11, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(11, 11, PointType.VERTEX)).toPathPoint(),
      (new PolyPathPoint2D(11, 12, PointType.CURVE)).toPathPoint(),
      (new PolyPathPoint2D(10, 12, PointType.CURVE)).toPathPoint()],
    true
  ))

  let state = getState()
  let labels = state.task.items[0].labels
  const labelIds = Object.keys(labels)
  expect(labelIds.length).toEqual(3)
  expect(labels[labelIds[0]].order).toEqual(0)
  expect(labels[labelIds[1]].order).toEqual(1)
  expect(labels[labelIds[2]].order).toEqual(2)

  // Move last label back
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[2]))

  keyClick(label2dHandler, Key.ARROW_DOWN)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(0)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(1)

  // Move first label forward
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[0]))
  keyClick(label2dHandler, Key.ARROW_UP)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(1)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(0)

  // Move label in front to back
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  keyClick(label2dHandler, Key.B_LOW)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(2)
  expect(labels[labelIds[1]].order).toEqual(0)
  expect(labels[labelIds[2]].order).toEqual(1)

  // Move label in back to front
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  keyClick(label2dHandler, Key.F_LOW)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(1)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(0)
})
