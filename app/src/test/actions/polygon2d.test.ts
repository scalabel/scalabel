import _ from 'lodash'
import * as action from '../../js/action/common'
import * as polygon2d from '../../js/action/polygon2d'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { LabelTypeName } from '../../js/common/types'
import { PathPoint2D, PointType } from '../../js/drawable/2d/path_point2d'
import { makePolygon } from '../../js/functional/states'
import { PolygonType } from '../../js/functional/types'
import { testJson } from '../test_image_objects'
import { checkPathPointFields } from '../util'

test('Add, change and delete polygon labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(polygon2d.addPolygon2dLabel(
    itemIndex, -1, [0],
    [(new PathPoint2D(0, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(1, 1, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(1, 2, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(0, 2, PointType.CURVE)).toPathPoint()],
    true
  ))
  Session.dispatch(polygon2d.addPolygon2dLabel(
    itemIndex, -1, [0],
    [(new PathPoint2D(3, 4, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(4, 4, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(4, 5, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(3, 5, PointType.CURVE)).toPathPoint()],
    false
  ))
  Session.dispatch(polygon2d.addPolygon2dLabel(
    itemIndex, -1, [0],
    [(new PathPoint2D(10, 11, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(11, 11, PointType.VERTEX)).toPathPoint(),
      (new PathPoint2D(11, 12, PointType.CURVE)).toPathPoint(),
      (new PathPoint2D(10, 12, PointType.CURVE)).toPathPoint()],
    true, false
  ))
  let state = Session.getState()
  let item = state.task.items[itemIndex]
  const labels = item.labels
  const shapes = item.shapes
  expect(_.size(labels)).toBe(3)
  expect(_.size(shapes)).toBe(3)

  const labelIds: number[] = _.map(labels, (l) => l.id)
  let label = labels[labelIds[0]]
  expect(label.item).toBe(itemIndex)
  expect(label.type).toBe(LabelTypeName.POLYGON_2D)
  expect(label.manual).toBe(true)
  expect(labels[labelIds[2]].manual).toBe(false)

  const indexedShape = shapes[label.shapes[0]]
  let shape = indexedShape.shape as PolygonType
  // Check label ids
  let index = 0
  _.forEach(labels, (v, i) => {
    expect(v.id).toBe(Number(i))
    expect(v.id).toBe(index)
    index += 1
  })
  // Check shape ids
  index = 0
  _.forEach(shapes, (v, i) => {
    expect(v.id).toBe(Number(i))
    expect(v.id).toBe(index)
    index += 1
  })

  let points = shape.points

  checkPathPointFields(points[0], 0, 1, true)
  checkPathPointFields(points[1], 1, 1, true)
  checkPathPointFields(points[2], 1, 2, false)
  checkPathPointFields(points[3], 0, 2, false)

  Session.dispatch(
    action.changeLabelShape(
      itemIndex, indexedShape.id, makePolygon({ points:
      [(new PathPoint2D(2, 0, PointType.CURVE)).toPathPoint(),
        (new PathPoint2D(4, 0, PointType.CURVE)).toPathPoint(),
        (new PathPoint2D(4, 2, PointType.VERTEX)).toPathPoint(),
        (new PathPoint2D(2, 2, PointType.CURVE)).toPathPoint()]})))

  state = Session.getState()
  item = state.task.items[itemIndex]
  label = item.labels[label.id]
  shape = item.shapes[label.shapes[0]].shape as PolygonType

  points = shape.points

  checkPathPointFields(points[0], 2, 0, false)
  checkPathPointFields(points[1], 4, 0, false)
  checkPathPointFields(points[2], 4, 2, true)
  checkPathPointFields(points[3], 2, 2, false)

  Session.dispatch(action.deleteLabel(itemIndex, label.id))
  state = Session.getState()
  expect(_.size(state.task.items[itemIndex].labels)).toBe(2)
  expect(_.size(state.task.items[itemIndex].shapes)).toBe(2)
})
