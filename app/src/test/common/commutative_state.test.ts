import _ from 'lodash'
import * as box2d from '../../js/action/box2d'
import * as action from '../../js/action/common'
import { configureStore } from '../../js/common/configure_store'
import { testJson } from '../test_states/test_image_objects'

describe('Test actions are commutative (for the task)', () => {
  test('Add, change and delete box2d labels', () => {
    const storeA = configureStore(testJson)
    const storeB = configureStore(testJson)
    const itemIndex = 0
    storeA.dispatch(action.goToItem(itemIndex))
    storeB.dispatch(action.goToItem(itemIndex))

    const actionA = box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4)
    const actionB = box2d.addBox2dLabel(itemIndex, -1, [0], {}, 5, 6, 7, 8)

    storeA.dispatch(actionA)
    storeA.dispatch(actionB)

    storeB.dispatch(actionB)
    storeB.dispatch(actionA)

    const stateA = storeA.getState().present.task
    const stateB = storeB.getState().present.task

    console.log(stateA.items[0].labels)
    console.log(stateB.items[0].labels)
    expect(stateA).toStrictEqual(stateB)
  })

})
