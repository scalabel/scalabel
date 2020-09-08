import _ from "lodash"

import { addPolygon2dLabel } from "../../../../src/action/polygon2d"
import { TrackInterp } from "../../../../src/auto/track/interp/interp"
import { Points2DLinearInterp } from "../../../../src/auto/track/interp/linear/points2d"
import { dispatch, getState } from "../../../../src/common/session"
import {
  LabelType,
  PathPoint2DType,
  PathPointType,
  SimplePathPoint2DType
} from "../../../../src/types/state"
import { setupTestStore } from "../../../components/util"
import { emptyTrackingTask } from "../../../test_states/test_track_objects"

beforeEach(() => {
  setupTestStore(emptyTrackingTask)
})

test("2D polygon linear interpolation", () => {
  const interp: TrackInterp = new Points2DLinearInterp()
  const points: SimplePathPoint2DType[] = [
    { x: 10, y: 11, pointType: PathPointType.LINE },
    { x: 10, y: 21, pointType: PathPointType.LINE },
    { x: 30, y: 11, pointType: PathPointType.LINE }
  ]

  // Making some labels and shapes
  for (let i = 0; i < 8; ++i) {
    dispatch(addPolygon2dLabel(i, -1, [], points, true))
  }
  const state = getState()
  const labels = state.task.items.map((item) =>
    _.sample(item.labels)
  ) as LabelType[]
  let shapes = state.task.items.map((item, index) =>
    labels[index].shapes.map((s) => item.shapes[s])
  ) as PathPoint2DType[][]

  // Change the middle label
  const newLabel = _.cloneDeep(labels[4])
  let newShape = _.cloneDeep(shapes[4])
  labels.map((l) => {
    l.manual = false
  })
  labels[1].manual = true
  labels[6].manual = true

  const newPolygon: SimplePathPoint2DType[] = [
    { x: 22, y: 35, pointType: PathPointType.LINE },
    { x: 22, y: 45, pointType: PathPointType.LINE },
    { x: 42, y: 35, pointType: PathPointType.LINE }
  ]
  newShape = newShape.map((s, i) => ({ ...s, ...newPolygon[i] }))

  shapes = interp.interp(
    newLabel,
    newShape,
    labels,
    shapes
  ) as PathPoint2DType[][]
  expect(shapes[4]).toMatchPoints2D(newPolygon)
  // Only update between manual labels
  expect(shapes[0]).toMatchPoints2D(points)
  expect(shapes[7]).toMatchPoints2D(points)
  expect(shapes[1]).toMatchPoints2D(points)
  expect(shapes[6]).toMatchPoints2D(points)
  // Interpolation results
  expect(shapes[2]).toMatchPoints2D([
    { x: 14, y: 19 },
    { x: 14, y: 29 },
    { x: 34, y: 19 }
  ])
  expect(shapes[3]).toMatchPoints2D([
    { x: 18, y: 27 },
    { x: 18, y: 37 },
    { x: 38, y: 27 }
  ])
  expect(shapes[5]).toMatchPoints2D([
    { x: 16, y: 23 },
    { x: 16, y: 33 },
    { x: 36, y: 23 }
  ])
})
