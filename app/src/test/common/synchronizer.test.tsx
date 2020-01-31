import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, render } from '@testing-library/react'
import React from 'react'
import Dashboard, {
  Synchronizer
} from '../../js/common/synchronizer'
import { myTheme } from '../../js/styles/theme'

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test sending until acked', () => {
    const taskIndex = 0
    const projectName = 'testProject'
    const callback = () => {}
    const synchronizer = new Synchronizer(taskIndex, projectName, callback)
    return 0
  })
})
