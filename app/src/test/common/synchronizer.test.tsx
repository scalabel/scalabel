import { cleanup } from '@testing-library/react'
import { Synchronizer } from '../../js/common/synchronizer'

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test sending until acked', () => {
    const taskIndex = 0
    const projectName = 'testProject'
    const userId = 'user'
    const callback = () => {}
    const synchronizer = new Synchronizer(
      taskIndex, projectName, userId, callback)
    return 0
  })
})
