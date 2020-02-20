import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, fireEvent, render } from '@testing-library/react'
import * as React from 'react'
import io from 'socket.io-client'
import { addLabel } from '../../js/action/common'
import { ActionType, SUBMIT } from '../../js/action/types'
import Session, { ConnectionStatus } from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Synchronizer } from '../../js/common/synchronizer'
import TitleBar from '../../js/components/title_bar'
import { makeLabel } from '../../js/functional/states'
import { EventName } from '../../js/server/types'
import { myTheme } from '../../js/styles/theme'
import { testJson } from '../test_image_objects'

beforeEach(() => {
  cleanup()
})
afterEach(cleanup)

describe('Save button functionality', () => {
  test('Autosave on: no save button', async () => {
    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    io.connect = jest.fn().mockImplementation(() => mockSocket)
    const synchronizer = new Synchronizer(0, 'test', 'fakeId', () => { return })
    initStore(testJson, synchronizer.middleware)
    Session.autosave = true

    // add a fake task action to be saved
    synchronizer.actionQueue.push(addLabel(0, makeLabel()))

    // only need to test save button for manual saving
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave={Session.autosave}
          synchronizer={synchronizer}
        />
      </MuiThemeProvider>
    )
    // Autosave on -> no save button
    expect(() => { getByTestId('Save') }).toThrow(Error)
  })

  test('Autosave off: save button saves and updates status', async () => {
    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    io.connect = jest.fn().mockImplementation(() => mockSocket)
    const synchronizer = new Synchronizer(0, 'test', 'fakeId', () => { return })
    initStore(testJson, synchronizer.middleware)
    Session.autosave = false

    // add a fake task action to be saved
    synchronizer.actionQueue.push(addLabel(0, makeLabel()))

    // only need to test save button for manual saving
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave={Session.autosave}
          synchronizer={synchronizer}
        />
      </MuiThemeProvider>
    )
    const saveButton = getByTestId('Save')
    fireEvent.click(saveButton)
    expect(Session.status).toBe(ConnectionStatus.SAVING)
    expect(mockSocket.emit).toHaveBeenCalled()
  })
})

describe('Submit button functionality', () => {
  test('Autosave on: submit button just updates flag', async () => {
    Session.autosave = true

    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    io.connect = jest.fn().mockImplementation(() => mockSocket)
    const synchronizer = new Synchronizer(0, 'test', 'fakeId', () => { return })
    initStore(testJson, synchronizer.middleware)

    // only need to test save button for manual saving
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave={Session.autosave}
          synchronizer={synchronizer}
        />
      </MuiThemeProvider>
    )

    const dispatchSpy = jest.spyOn(Session, 'dispatch')
    const submitButton = getByTestId('Submit')
    fireEvent.click(submitButton)
    checkSubmitDispatch(dispatchSpy)

    // autosave is on, so the dispatch will trigger a save
    expect(mockSocket.emit).toHaveBeenCalled()
    const emitCalls = mockSocket.emit.mock.calls

    // 2 calls: 1 for init session, 1 for submit
    expect(emitCalls.length).toBe(2)

    // each call has 2 args: event name and action message
    const emitArgs = emitCalls[1]
    expect(emitArgs.length).toBe(2)
    expect(emitArgs[0]).toBe(EventName.ACTION_SEND)

    // the action message should just contain the submit action
    const actionMessage = JSON.parse(emitArgs[1])
    expect(actionMessage.actions.length).toBe(1)
    expect(actionMessage.actions[0].type).toBe(SUBMIT)
  })

  test('Autosave off: submit button updates flag and saves', async () => {
    Session.autosave = false

    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    io.connect = jest.fn().mockImplementation(() => mockSocket)
    const synchronizer = new Synchronizer(0, 'test', 'fakeId', () => { return })
    initStore(testJson, synchronizer.middleware)

    // only need to test save button for manual saving
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave={Session.autosave}
          synchronizer={synchronizer}
        />
      </MuiThemeProvider>
    )
    const dispatchSpy = jest.spyOn(Session, 'dispatch')
    const submitButton = getByTestId('Submit')
    fireEvent.click(submitButton)

    // check that submit action was dispatched
    checkSubmitDispatch(dispatchSpy)

    // make sure that the submit action was saved
    expect(mockSocket.emit).toHaveBeenCalled()
    const emitCalls = mockSocket.emit.mock.calls

    // 1 call: initSession and submit actions are saved together
    expect(emitCalls.length).toBe(1)

    // each call has 2 args: event name and action message
    const emitArgs = emitCalls[0]
    expect(emitArgs.length).toBe(2)
    expect(emitArgs[0]).toBe(EventName.ACTION_SEND)
    const actionMessage = JSON.parse(emitArgs[1])
    expect(actionMessage.actions.length).toBe(2)
    // the first action should be INIT_SESSION, so the second is submit
    expect(actionMessage.actions[1].type).toBe(SUBMIT)
  })
})

/**
 * Checks that submit action was dispatched
 */
function checkSubmitDispatch
  (dispatchSpy: jest.SpyInstance<void, [ActionType]>) {
  expect(dispatchSpy).toHaveBeenCalled()
  // Check type, instead of HaveBeenCalledWith, because userId may change
  const dispatchCalls = dispatchSpy.mock.calls
  const dispatchAction = dispatchCalls[dispatchCalls.length - 1][0]
  expect(dispatchAction.type).toBe(SUBMIT)
}
