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

afterAll(() => {
  windowInterface.XMLHttpRequest = oldXMLHttpRequest
  windowInterface.alert = oldAlert
})

interface WindowInterface extends Window {
  /** XML request init function */
  XMLHttpRequest: () => {
    /** Callback function for xhr request */
    onreadystatechange: () => void
  }
}

const windowInterface = window as WindowInterface
const oldXMLHttpRequest = windowInterface.XMLHttpRequest
const oldAlert = windowInterface.alert

const xhrMockClass = {
  open: jest.fn(),
  send: jest.fn(),
  onreadystatechange: jest.fn(),
  readyState: 4,
  response: JSON.stringify(0)
}
windowInterface.XMLHttpRequest = jest.fn(() => xhrMockClass)
windowInterface.alert = jest.fn()

describe('Save button functionality', () => {
  let saveButton: HTMLElement

  beforeEach(() => {
    initStore({ testKey: 'testValue' })
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
      expect.stringContaining('"testKey":"testValue"')
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
    expect(windowInterface.alert).toBeCalled()
    expect(Session.status).toBe(ConnectionStatus.SAVED)

    xhrMockClass.response = JSON.stringify(0)
  })
})
