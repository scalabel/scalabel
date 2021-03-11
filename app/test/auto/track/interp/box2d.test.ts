import _ from "lodash"

import { addBox2dLabel } from "../../../../src/action/box2d"
import { TrackInterp } from "../../../../src/auto/track/interp/interp"
import { Box2DLinearInterp } from "../../../../src/auto/track/interp/linear/box2d"
import { dispatch, getState } from "../../../../src/common/session"
import { LabelType, RectType, SimpleRect } from "../../../../src/types/state"
import { setupTestStore } from "../../../components/util"
import { emptyTrackingTask } from "../../../test_states/test_track_objects"

beforeEach(() => {
  setupTestStore(emptyTrackingTask)
})

test("2D box linear interpolation", () => {
  const interp: TrackInterp = new Box2DLinearInterp()
  const box: SimpleRect = { x1: 0, y1: 0, x2: 20, y2: 20 }

  // Making some labels and shapes
  for (let i = 0; i < 8; ++i) {
    dispatch(addBox2dLabel(i, -1, [], {}, box))
  }
  const state = getState()
  const labels = state.task.items.map((item) =>
    _.sample(item.labels)
  ) as LabelType[]
  let shapes = state.task.items.map((item, index) => [
    item.shapes[labels[index].shapes[0]]
  ]) as RectType[][]

  // Change the middle label
  const newLabel = _.cloneDeep(labels[4])
  const newShape = _.cloneDeep(shapes[4])
  labels.map((l) => {
    l.manual = false
  })
  labels[1].manual = true
  labels[6].manual = true

  const newBox: SimpleRect = { x1: 36, y1: 72, x2: 32, y2: 44 }
  newShape[0] = { ...newShape[0], ...newBox }

  shapes = interp.interp(newLabel, newShape, labels, shapes) as RectType[][]
  expect(shapes[4][0]).toMatchObject(newBox)
  // Only update between manual labels
  expect(shapes[0][0]).toMatchObject(box)
  expect(shapes[7][0]).toMatchObject(box)
  expect(shapes[1][0]).toMatchObject(box)
  expect(shapes[6][0]).toMatchObject(box)
  // Interpolation results
  expect(shapes[2][0]).toMatchObject({ x1: 12, y1: 24, x2: 24, y2: 28 })
  expect(shapes[3][0]).toMatchObject({ x1: 24, y1: 48, x2: 28, y2: 36 })
  expect(shapes[5][0]).toMatchObject({ x1: 18, y1: 36, x2: 26, y2: 32 })
})
