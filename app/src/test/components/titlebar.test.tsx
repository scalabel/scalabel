import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, fireEvent, render } from '@testing-library/react'
import * as React from 'react'
import io from 'socket.io-client'
import { addLabel } from '../../js/action/common'
import Session, { ConnectionStatus } from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import TitleBar from '../../js/components/title_bar'
import { makeLabel } from '../../js/functional/states'
import { myTheme } from '../../js/styles/theme'

beforeEach(() => {
  cleanup()
})
afterEach(cleanup)

describe('Save button functionality', () => {
  test('Save button triggers save and updates status', async () => {
    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    io.connect = jest.fn().mockImplementation(() => mockSocket)
    const synchronizer = new Synchronizer(0, 'test', () => { return })
    // add a fake task action to be saved
    synchronizer.actionQueue.push(addLabel(0, makeLabel()))

    // only need to test save button for manual saving
    Session.autosave = false
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave={false}
          synchronizer={synchronizer}
        />
      </MuiThemeProvider>
    )
    const saveButton = getByTestId('Save')
    expect(Session.status).toBe(ConnectionStatus.UNSAVED)
    fireEvent.click(saveButton)
    expect(Session.status).toBe(ConnectionStatus.SAVING)
    expect(mockSocket.emit).toHaveBeenCalled()
  })
})
