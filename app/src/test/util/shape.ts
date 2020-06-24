import { getState } from '../../js/common/session'
import { GetStateFunc } from '../../js/common/simple_store'
import { getShape } from '../../js/functional/state_util'
import { IdType, PolygonType, SimpleRect } from '../../js/functional/types'

/**
 * Check that the box's coords are correct
 * @param labelId: Id of the box
 * @param coords: Coords of the box
 * @param itemIndex: item index of the label. Use the current item by default
 * @param getStateFunc: get the state. Use the session get state by default
 */
export function checkBox2D (
  labelId: IdType, target: Partial<SimpleRect>,
  itemIndex: number = -1,
  getStateFunc: GetStateFunc = getState) {
  const state = getStateFunc()
  if (itemIndex < 0) {
    itemIndex = state.user.select.item
  }
  const rect = getShape(state, itemIndex, labelId, 0)
  expect(rect).toMatchObject(target)
}

/**
 * Check that the polygon's vertices are correct
 * @param labelId: Id of the polygon
 * @param coords: List of vertices of the polygon
 * @param itemIndex: item index of the label. Use the current item by default
 * @param getStateFunc: get the state. Use the session get state by default
 */
export function checkPolygon (
  labelId: IdType, coords: number[][],
  itemIndex: number = -1,
  getStateFunc: GetStateFunc = getState) {
  const state = getStateFunc()
  if (itemIndex < 0) {
    itemIndex = state.user.select.item
  }
  const points = (getShape(state, 0, labelId, 0) as PolygonType).points
  expect(points.length).toEqual(coords.length)
  for (let pointIndex = 0; pointIndex < points.length; pointIndex++) {
    expect(points[pointIndex]).toMatchObject(
      { x: coords[pointIndex][0],
        y: coords[pointIndex][1],
        pointType: 'vertex' })
  }
}
