import _ from 'lodash'
import * as box2d from '../../js/action/box2d'
import * as action from '../../js/action/common'
import { BaseAction } from '../../js/action/types'
import { configureStore, FullStore } from '../../js/common/configure_store'
import { makeItem, makeTask } from '../../js/functional/states'
import { RectType } from '../../js/functional/types'

const itemIndex = 0

describe('Test actions are commutative (for the task)', () => {
  test('Adding two boxes', () => {
    const [storeA, storeB] = makeStores()

    // Add two separate 2d bbox labels
    const actionA = box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4)
    const actionB = box2d.addBox2dLabel(itemIndex, -1, [0], {}, 5, 6, 7, 8)
    commutativeDispatch(storeA, storeB, actionA, actionB)

    // Check tasks are equal, and have two labels/shapes each
    checkTasksEqual(storeA, storeB)
    const item = storeA.getState().present.task.items[itemIndex]
    expect(_.size(item.labels)).toBe(2)
    expect(_.size(item.shapes)).toBe(2)
  })

  test('Changing the same shape, when change is independent', () => {
    const [storeA, storeB, , shapeId] = makeStoresWithBox()

    // Make two independent changes to the 2d box
    const actionA = box2d.changeBox2d(itemIndex, shapeId, { x1: 0, y1: 0 })
    const actionB = box2d.changeBox2d(itemIndex, shapeId, { x2: 10, y2: 10 })
    commutativeDispatch(storeA, storeB, actionA, actionB)

    // Check tasks are equal, and have the correct shape
    checkTasksEqual(storeA, storeB)
    const item = storeA.getState().present.task.items[itemIndex]
    const rect = item.shapes[shapeId] as RectType
    expect(rect.x1).toBe(0)
    expect(rect.y1).toBe(0)
    expect(rect.x2).toBe(10)
    expect(rect.y2).toBe(10)
  })

  test('Deleting a box twice causes no errors', () => {
    const [storeA, storeB, labelId] = makeStoresWithBox()

    // Delete twice for each store
    const deleteAction = action.deleteLabel(itemIndex, labelId)
    commutativeDispatch(storeA, storeB, deleteAction, deleteAction)

    // Check tasks are equal, and contain no labels or shapes
    checkTasksEqual(storeA, storeB)
    const item = storeA.getState().present.task.items[itemIndex]
    expect(_.size(item.labels)).toBe(0)
    expect(_.size(item.shapes)).toBe(0)
  })

  test(`Deleting and changing a shape
    results in deletion with no errors`, () => {
    const [storeA, storeB, labelId, shapeId] = makeStoresWithBox()

    const deleteAction = action.deleteLabel(itemIndex, labelId)
    const changeAction = box2d.changeBox2d(itemIndex, shapeId, { x1: 0, y1: 0 })

    commutativeDispatch(storeA, storeB, deleteAction, changeAction)

    // Check tasks are equal, and neither has any labels or shapes
    checkTasksEqual(storeA, storeB)
    const item = storeA.getState().present.task.items[itemIndex]
    expect(_.size(item.labels)).toBe(0)
    expect(_.size(item.shapes)).toBe(0)
  })
})

/**
 * Make two identical stores with 1st item selected
 */
function makeStores (): [FullStore, FullStore] {
  const task = makeTask({
    items: [makeItem()]
  })
  const storeA = configureStore({ task })
  const storeB = configureStore({ task })

  storeA.dispatch(action.goToItem(itemIndex))
  storeB.dispatch(action.goToItem(itemIndex))

  return [storeA, storeB]
}

/**
 * Make two identical stores with a single 2d box label
 */
function makeStoresWithBox (): [FullStore, FullStore, string, string] {
  const [storeA, storeB] = makeStores()

  const addBox = box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4)
  storeA.dispatch(addBox)
  storeB.dispatch(addBox)
  const labelId = addBox.labels[0][0].id
  const shapeId = addBox.shapes[0][0][0].id

  return [storeA, storeB, labelId, shapeId]
}

/**
 * Dispatch actions to stores in opposite order
 */
function commutativeDispatch (
  storeA: FullStore, storeB: FullStore,
  actionA: BaseAction, actionB: BaseAction) {
  storeA.dispatch(actionA)
  storeA.dispatch(actionB)

  storeB.dispatch(actionB)
  storeB.dispatch(actionA)
}

/**
 * Check tasks are equal
 */
function checkTasksEqual (storeA: FullStore, storeB: FullStore) {
  const stateA = storeA.getState().present.task
  const stateB = storeB.getState().present.task
  expect(stateA).toStrictEqual(stateB)
}
