import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, fireEvent, render } from '@testing-library/react'
import * as React from 'react'
import Session, { ConnectionStatus } from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import TitleBar from '../../js/components/title_bar'
import { myTheme } from '../../js/styles/theme'

beforeEach(() => {
  Session.devMode = false
  jest.clearAllMocks()
})

afterEach(cleanup)

/* tslint:disable */
afterAll(() => {
  (window as any).XMLHttpRequest = oldXMLHttpRequest
  window.alert = oldAlert
})

const oldXMLHttpRequest = (window as any).XMLHttpRequest
const oldAlert = window.alert

const xhrMockClass = {
  open: jest.fn(),
  send: jest.fn(),
  onreadystatechange: jest.fn(),
  readyState: 4,
  response: JSON.stringify(0)
};
(window as any).XMLHttpRequest =
  jest.fn().mockImplementation(() => xhrMockClass)
window.alert = jest.fn()
/* tslint:enable */

describe('Save button functionality', () => {
  let saveButton: HTMLElement

  beforeEach(() => {
    initStore({
      task: {
        taskTestKey: 'taskTestValue'
      },
      user: {
        userTestKey: 'userTestValue'
      },
      session: {
        sessionTestKey: 'sessionTestValue'
      }
    })
    jest.clearAllMocks()
    // recreate for each test because of timeout side effects
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
        />
      </MuiThemeProvider>
    )
    saveButton = getByTestId('Save')
  })

  test('Save button triggers save', () => {
    fireEvent.click(saveButton)

    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalledWith(
      expect.stringContaining('"taskTestKey":"taskTestValue"')
    )
    expect(xhrMockClass.send).toBeCalledWith(
      expect.stringContaining('"userTestKey":"userTestValue"')
    )
    expect(xhrMockClass.send).toBeCalledWith(
      expect.stringContaining('"sessionTestKey":"sessionTestValue"')
    )
    xhrMockClass.onreadystatechange()
  })

  test('Sync status is correct during save', () => {
    expect(Session.status).toBe(ConnectionStatus.SAVED)
    fireEvent.click(saveButton)

    expect(Session.status).toBe(ConnectionStatus.SAVING)
    xhrMockClass.onreadystatechange()
    expect(Session.status).toBe(ConnectionStatus.SAVED)
  })

  test('Alerts on save failure', () => {
    xhrMockClass.response = JSON.stringify(null)
    fireEvent.click(saveButton)

    xhrMockClass.onreadystatechange()
    expect(window.alert).toBeCalled()
    expect(Session.status).toBe(ConnectionStatus.UNSAVED)

    xhrMockClass.response = JSON.stringify(0)
  })
})
