import _ from "lodash"

import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import { ShapeTypeName } from "../../src/const/common"
import { Label2DHandler } from "../../src/drawable/2d/label2d_handler"
import { makeImageViewerConfig } from "../../src/functional/states"
import { Size2D } from "../../src/math/size2d"
import { Vector2D } from "../../src/math/vector2d"
import { IdType, Point2DType } from "../../src/types/state"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_image_objects"
import { findNewLabels } from "../util/state"
import { mouseDown, mouseMove, mouseUp } from "./util"

/**
 * Initialize Session, label 2d list, label 2d handler
 */
function initializeTestingObjects(): [Label2DHandler, number] {
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

test.skip("Draw human pose", () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(action.changeSelect({ labelType: 4 }))
  const labelIds: IdType[] = []

  const canvasSize = new Size2D(1000, 1000)
  mouseMove(label2dHandler, 100, 100, canvasSize, -1, 0)
  mouseDown(label2dHandler, 100, 100, -1, 0)
  mouseMove(label2dHandler, 200, 200, canvasSize, -1, 0)
  mouseUp(label2dHandler, 200, 200, -1, 0)
  labelIds.push(findNewLabels(getState().task.items[0].labels, labelIds)[0])

  const state = getState()

  const spec =
    state.task.config.label2DTemplates[
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
    bottomRight.x - upperLeft.x,
    bottomRight.y - upperLeft.y
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
      ((templatePoint.x - upperLeft.x) / dimensions.x) * 100 + 100
    )
    expect(point.y).toBeCloseTo(
      ((templatePoint.y - upperLeft.y) / dimensions.y) * 100 + 100
    )
  }
})
