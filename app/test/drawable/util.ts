import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import { Label2DHandler } from "../../src/drawable/2d/label2d_handler"
import { makeImageViewerConfig } from "../../src/functional/states"
import { Size2D } from "../../src/math/size2d"
import { Vector2D } from "../../src/math/vector2d"
import { IdType, INVALID_ID, SimpleRect } from "../../src/types/state"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_image_objects"
import { LabelCollector } from "../util/label_collector"

/**
 * Create a polygon by clicking at each point in sequence
 *
 * @param label2dHandler
 * @param canvasSize
 * @param points: the input vertices
 * @param interrupt: add an interrupt saving action
 * @param valid: whether the polygon is valid
 * @param points
 * @param interrupt
 * @param valid
 */
export function drawPolygon(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  points: number[][],
  interrupt: boolean = false,
  valid: boolean = true
): IdType {
  const labelIds = new LabelCollector(getState)
  const start = labelIds.collect()

  for (const p of points) {
    mouseMoveClick(label2dHandler, p[0], p[1], canvasSize, -1, 0)
    if (interrupt) {
      dispatch(action.setStatusToSaved())
    }
  }

  // Handler index of 1 marks the end of the polygon
  mouseMoveClick(label2dHandler, points[0][0], points[0][1], canvasSize, -1, 1)

  const total = labelIds.collect()
  if (valid) {
    expect(labelIds.length).toEqual(start + 1)
  }
  if (total > start) {
    return labelIds[total - 1]
  } else {
    return INVALID_ID
  }
}

/**
 * Create a polygon by dragging between each pair of subsequent points
 *
 * @param label2dHandler
 * @param canvasSize
 * @param points: the input vertices
 * @param interrupt: add an interrupt saving action
 * @param valid: whether the polygon is valid
 * @param points
 * @param interrupt
 * @param valid
 */
export function drawPolygonByDragging(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  points: number[][],
  interrupt: boolean = false,
  valid: boolean = true
): IdType {
  const labelIds = new LabelCollector(getState)
  const start = labelIds.collect()

  // Start by clicking at the first point
  mouseMoveClick(label2dHandler, points[0][0], points[0][1], canvasSize, -1, 0)

  // Then press down to start dragging
  mouseMove(label2dHandler, points[0][0], points[0][1], canvasSize, -1, 0)
  mouseDown(label2dHandler, points[0][0], points[0][1], -1, 0)

  for (const p of points.slice(1)) {
    // Drag to the next point before lifting mouse
    mouseMove(label2dHandler, p[0], p[1], canvasSize, -1, 0)
    mouseUp(label2dHandler, p[0], p[1], -1, 0)

    if (interrupt) {
      dispatch(action.setStatusToSaved())
    }

    // Click to prepare for the next line
    mouseDown(label2dHandler, p[0], p[1], -1, 0)
  }

  // Lift up at the first point
  // Handler index of 1 marks the end of the polygon
  mouseMove(label2dHandler, points[0][0], points[0][1], canvasSize, -1, 1)
  mouseUp(label2dHandler, points[0][0], points[0][1], -1, 1)

  const total = labelIds.collect()
  if (valid) {
    expect(labelIds.length).toEqual(start + 1)
  }
  if (total > start) {
    return labelIds[total - 1]
  } else {
    return INVALID_ID
  }
}

/**
 * Make mouse movements to do some 2d box operation
 *
 * @param coords: coords for starting and ending clicks
 * @param labelIndex: if not -1, specifies changing an existing label
 * @param interrupt: if enabled, interrupt the operation with another action
 * @param label2dHandler
 * @param canvasSize
 * @param coords
 * @param labelIndex
 * @param handleIndex
 * @param interrupt
 */
function doBox2DOperation(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  coords: SimpleRect,
  labelIndex: number,
  handleIndex: number,
  interrupt: boolean = false
): void {
  const x1 = coords.x1
  const x2 = coords.x2
  const y1 = coords.y1
  const y2 = coords.y2

  // First point
  mouseMove(label2dHandler, x1, y1, canvasSize, labelIndex, handleIndex)
  mouseDown(label2dHandler, x1, y1, labelIndex, handleIndex)

  // Some random intermediate move
  mouseMove(
    label2dHandler,
    x1 + Math.random() * 5,
    y1 - Math.random() * 5,
    canvasSize,
    -1,
    0
  )

  if (interrupt) {
    dispatch(action.setStatusToSaved())
  }

  // Last point
  mouseMove(label2dHandler, x2, y2, canvasSize, -1, 0)
  // Mouse up doesn't need to be exactly at the point
  mouseUp(label2dHandler, x2 + Math.random(), y2 - Math.random(), -1, 0)
}

/**
 * Make mouse movements to add the 2D box
 *
 * @param coords: the coordinates of the new box
 * @param label2dHandler
 * @param canvasSize
 * @param coords
 * @param interrupt
 */
export function drawBox2D(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  coords: SimpleRect,
  interrupt: boolean = false
): IdType {
  const boxIds = new LabelCollector(getState)
  const start = boxIds.collect()
  doBox2DOperation(label2dHandler, canvasSize, coords, -1, 0, interrupt)
  boxIds.collect()
  expect(boxIds.length).toEqual(start + 1)
  return boxIds[start]
}

/**
 * Make mouse movements to resize the 2D box
 *
 * @param coords: x1/y1 is the existing point, x2/y2 is the new point
 * @param labelIndex: the index of the label to move
 * @param label2dHandler
 * @param canvasSize
 * @param coords
 * @param labelIndex
 * @param interrupt
 */
export function resizeBox2D(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  coords: SimpleRect,
  labelIndex: number,
  interrupt: boolean = false
): void {
  doBox2DOperation(
    label2dHandler,
    canvasSize,
    coords,
    labelIndex,
    labelIndex,
    interrupt
  )
}

/**
 * Make mouse movements to move the 2D box
 *
 * @param coords: x1/y1 is the existing point, x2/y2 is the new point
 * @param labelIndex: the index of the label to move
 * @param label2dHandler
 * @param canvasSize
 * @param coords
 * @param labelIndex
 * @param interrupt
 */
export function moveBox2D(
  label2dHandler: Label2DHandler,
  canvasSize: Size2D,
  coords: SimpleRect,
  labelIndex: number,
  interrupt: boolean = false
): void {
  // Handle index of 0 represents a move instead of a resize
  doBox2DOperation(label2dHandler, canvasSize, coords, labelIndex, 0, interrupt)
}

/**
 * Driver function for mouse move
 *
 * @param label2d
 * @param x
 * @param y
 * @param canvasSize
 * @param labelIndex
 * @param handleIndex
 */
export function mouseMove(
  label2d: Label2DHandler,
  x: number,
  y: number,
  canvasSize: Size2D,
  labelIndex: number,
  handleIndex: number
): void {
  label2d.onMouseMove(new Vector2D(x, y), canvasSize, labelIndex, handleIndex)
}

/**
 * Driver function for mouse down
 *
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseDown(
  label2d: Label2DHandler,
  x: number,
  y: number,
  labelIndex: number,
  handleIndex: number
): void {
  label2d.onMouseDown(new Vector2D(x, y), labelIndex, handleIndex)
}

/**
 * Driver function for mouse up
 *
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseUp(
  label2d: Label2DHandler,
  x: number,
  y: number,
  labelIndex: number,
  handleIndex: number
): void {
  label2d.onMouseUp(new Vector2D(x, y), labelIndex, handleIndex)
}

/**
 * Driver function for mouse click
 *
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseClick(
  label2d: Label2DHandler,
  x: number,
  y: number,
  labelIndex: number,
  handleIndex: number
): void {
  mouseDown(label2d, x, y, labelIndex, handleIndex)
  mouseUp(label2d, x, y, labelIndex, handleIndex)
}

/**
 * Driver function for mouse move and click
 *
 * @param label2d
 * @param x
 * @param y
 * @param canvasSize
 * @param labelIndex
 * @param handleIndex
 */
export function mouseMoveClick(
  label2d: Label2DHandler,
  x: number,
  y: number,
  canvasSize: Size2D,
  labelIndex: number,
  handleIndex: number
): void {
  mouseMove(label2d, x, y, canvasSize, labelIndex, handleIndex)
  mouseClick(label2d, x, y, labelIndex, handleIndex)
}

/**
 * Driver function for key down
 *
 * @param label2d
 * @param key
 */
export function keyDown(label2d: Label2DHandler, key: string): void {
  label2d.onKeyDown(new KeyboardEvent("keydown", { key }))
}

/**
 * Driver function for key up
 *
 * @param label2d
 * @param key
 */
export function keyUp(label2d: Label2DHandler, key: string): void {
  label2d.onKeyUp(new KeyboardEvent("keyup", { key }))
}

/**
 * Driver function for key click
 *
 * @param label2d
 * @param key
 */
export function keyClick(label2d: Label2DHandler, key: string): void {
  keyDown(label2d, key)
  keyUp(label2d, key)
}

/**
 * Initialize Session, label 2d list, label 2d handler
 */
export function initializeTestingObjects(): [Label2DHandler, number] {
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
