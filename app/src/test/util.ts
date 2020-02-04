import { PathPoint2DType, RectType, State, Vector3Type } from '../js/functional/types'

/** Get polygon points */
export function getPolygonPoints (
  state: State, itemIndex: number, labelId: number
): PathPoint2DType[] {
  const item = state.task.items[itemIndex]
  return item.labels[labelId].shapes.map(
    (shapeId) => item.indexedShapes[shapeId].shape as PathPoint2DType
  )
}

/**
 * Check equality between two Vector3Type objects
 * @param v1
 * @param v2
 */
export function expectVector3TypesClose (
  v1: Vector3Type, v2: Vector3Type, num = 2
) {
  expect(v1.x).toBeCloseTo(v2.x, num)
  expect(v1.y).toBeCloseTo(v2.y, num)
  expect(v1.z).toBeCloseTo(v2.z, num)
}

/**
 * Check that rectangles are close
 */
export function expectRectTypesClose (
  r1: RectType, r2: RectType, num = 2
) {
  expect(r1.x1).toBeCloseTo(r2.x1, num)
  expect(r1.x2).toBeCloseTo(r2.x2, num)
  expect(r1.y1).toBeCloseTo(r2.y1, num)
  expect(r1.y2).toBeCloseTo(r2.y2, num)
}
