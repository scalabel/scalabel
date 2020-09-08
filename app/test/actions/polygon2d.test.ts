import _ from "lodash"

import * as action from "../../src/action/common"
import * as polygon2d from "../../src/action/polygon2d"
import { dispatch, getState } from "../../src/common/session"
import { LabelTypeName } from "../../src/const/common"
import { getShapes } from "../../src/functional/state_util"
import { makeSimplePathPoint2D } from "../../src/functional/states"
import { PathPoint2DType, PathPointType } from "../../src/types/state"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_image_objects"

test("Add, change and delete polygon labels", () => {
  setupTestStore(testJson)

  const itemIndex = 0
  dispatch(action.goToItem(itemIndex))
  dispatch(
    polygon2d.addPolygon2dLabel(
      itemIndex,
      -1,
      [0],
      [
        makeSimplePathPoint2D(0, 1, PathPointType.LINE),
        makeSimplePathPoint2D(1, 1, PathPointType.LINE),
        makeSimplePathPoint2D(1, 2, PathPointType.CURVE),
        makeSimplePathPoint2D(0, 2, PathPointType.CURVE)
      ],
      true
    )
  )
  dispatch(
    polygon2d.addPolygon2dLabel(
      itemIndex,
      -1,
      [0],
      [
        makeSimplePathPoint2D(3, 4, PathPointType.LINE),
        makeSimplePathPoint2D(4, 4, PathPointType.LINE),
        makeSimplePathPoint2D(4, 5, PathPointType.LINE),
        makeSimplePathPoint2D(3, 5, PathPointType.LINE)
      ],
      false
    )
  )
  dispatch(
    polygon2d.addPolygon2dLabel(
      itemIndex,
      -1,
      [0],
      [
        makeSimplePathPoint2D(10, 11, PathPointType.LINE),
        makeSimplePathPoint2D(11, 11, PathPointType.LINE),
        makeSimplePathPoint2D(11, 12, PathPointType.LINE),
        makeSimplePathPoint2D(10, 12, PathPointType.LINE)
      ],
      true,
      false
    )
  )
  let state = getState()
  let item = state.task.items[itemIndex]
  const labels = item.labels
  const shapes = item.shapes
  expect(_.size(labels)).toBe(3)
  expect(_.size(shapes)).toBe(12)

  const labelIds = _.map(labels, (l) => l.id)
  let label = labels[labelIds[0]]
  expect(label.item).toBe(itemIndex)
  expect(label.type).toBe(LabelTypeName.POLYGON_2D)
  expect(label.manual).toBe(true)
  expect(labels[labelIds[2]].manual).toBe(false)

  _.forEach(labels, (v, i) => {
    expect(v.id).toBe(i)
  })
  // Check shape ids
  _.forEach(shapes, (v, i) => {
    expect(v.id).toBe(i)
  })

  const labelId = labelIds[0]
  let points = getShapes(state, itemIndex, labelId) as PathPoint2DType[]
  expect(points[0]).toMatchObject({
    label: [labelId],
    x: 0,
    y: 1,
    pointType: "line"
  })
  expect(points[1]).toMatchObject({
    label: [labelId],
    x: 1,
    y: 1,
    pointType: "line"
  })
  expect(points[2]).toMatchObject({
    label: [labelId],
    x: 1,
    y: 2,
    pointType: "bezier"
  })
  expect(points[3]).toMatchObject({
    label: [labelId],
    x: 0,
    y: 2,
    pointType: "bezier"
  })

  dispatch(
    action.changeShapes(
      itemIndex,
      points.map((p) => p.id),
      [
        makeSimplePathPoint2D(2, 0, PathPointType.LINE),
        makeSimplePathPoint2D(4, 0, PathPointType.LINE),
        makeSimplePathPoint2D(4, 2, PathPointType.LINE),
        makeSimplePathPoint2D(2, 2, PathPointType.LINE)
      ]
    )
  )

  state = getState()
  item = state.task.items[itemIndex]
  label = item.labels[label.id]
  // Shape = item.shapes[label.shapes[0]] as PolygonType
  points = getShapes(state, itemIndex, labelId) as PathPoint2DType[]

  expect(points[0]).toMatchObject({
    label: [labelId],
    x: 2,
    y: 0,
    pointType: "line"
  })
  expect(points[1]).toMatchObject({
    label: [labelId],
    x: 4,
    y: 0,
    pointType: "line"
  })
  expect(points[2]).toMatchObject({
    label: [labelId],
    x: 4,
    y: 2,
    pointType: "line"
  })
  expect(points[3]).toMatchObject({
    label: [labelId],
    x: 2,
    y: 2,
    pointType: "line"
  })

  dispatch(action.deleteLabel(itemIndex, label.id))
  item = getState().task.items[itemIndex]
  expect(_.size(item.labels)).toBe(2)
  expect(_.size(item.shapes)).toBe(8)
})
