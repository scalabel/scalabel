import { ThemeProvider } from "@material-ui/core/styles"
import { cleanup, fireEvent, render } from "@testing-library/react"
import _ from "lodash"
import * as React from "react"
import { Provider } from "react-redux"
import { ThunkAction } from "redux-thunk"

import { addLabel } from "../../src/action/common"
import Session from "../../src/common/session"
import { Synchronizer } from "../../src/common/synchronizer"
import TitleBar from "../../src/components/title_bar"
import { SUBMIT } from "../../src/const/action"
import { EventName } from "../../src/const/connection"
import { isStatusSaving } from "../../src/functional/selector"
import { makeLabel } from "../../src/functional/states"
import { scalabelTheme } from "../../src/styles/theme"
import { ActionType } from "../../src/types/action"
import { SyncActionMessageType } from "../../src/types/message"
import { ReduxState } from "../../src/types/redux"
import { State } from "../../src/types/state"
import { testJson } from "../test_states/test_image_objects"
import { setupTestStore, setupTestStoreWithMiddleware } from "./util"

beforeEach(() => {
  cleanup()
})
afterEach(cleanup)

// Need a different reference so selectors don't cache results
const testJsonAutosave = _.cloneDeep(testJson) as State

describe("Save button functionality", () => {
  test("Autosave on: no save button", async () => {
    testJsonAutosave.task.config.autosave = true
    setupTestStore(testJsonAutosave)

    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Provider store={Session.store}>
          <TitleBar />
        </Provider>
      </ThemeProvider>
    )
    expect(() => {
      getByTestId("Save")
    }).toThrow(Error)
  })

  test("Autosave off: save button triggers save action", async () => {
    const mockSocket = {
      connected: true,
      emit: jest.fn()
    }
    const synchronizer = new Synchronizer(mockSocket, 0, "test", "fakeId")

    testJsonAutosave.task.config.autosave = false
    setupTestStoreWithMiddleware(testJsonAutosave, synchronizer)

    // Add a fake task action to be saved
    synchronizer.actionQueue.push(addLabel(0, makeLabel()))

    // Only need to test save button for manual saving
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Provider store={Session.store}>
          <TitleBar />
        </Provider>
      </ThemeProvider>
    )
    const saveButton = getByTestId("Save")
    fireEvent.click(saveButton)
    expect(isStatusSaving(Session.store.getState())).toBe(true)
    expect(mockSocket.emit).toHaveBeenCalled()
  })
})

describe("Submit button functionality", () => {
  test("Autosave on: submit button just updates flag", async () => {
    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    const synchronizer = new Synchronizer(mockSocket, 0, "test", "fakeId")

    testJsonAutosave.task.config.autosave = true
    setupTestStoreWithMiddleware(testJsonAutosave, synchronizer)

    // Only need to test save button for manual saving
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Provider store={Session.store}>
          <TitleBar />
        </Provider>
      </ThemeProvider>
    )

    const dispatchSpy = jest.spyOn(Session, "dispatch")
    dispatchSpy.mockClear()

    const submitButton = getByTestId("Submit")
    fireEvent.click(submitButton)
    checkSubmitDispatch(dispatchSpy)

    // Autosave is on, so the dispatch will trigger a save
    expect(mockSocket.emit).toHaveBeenCalled()
    const emitCalls = mockSocket.emit.mock.calls

    // 2 calls: 1 for init session, 1 for submit
    expect(emitCalls.length).toBe(2)

    // Each call has 2 args: event name and action message
    const emitArgs = emitCalls[1]
    expect(emitArgs.length).toBe(2)
    expect(emitArgs[0]).toBe(EventName.ACTION_SEND)

    // The action message should just contain the submit action
    const actionMessage: SyncActionMessageType = emitArgs[1]
    const actionPacket = actionMessage.actions

    expect(actionPacket.actions.length).toBe(1)
    expect(actionPacket.actions[0].type).toBe(SUBMIT)
  })

  test("Autosave off: submit button updates flag and saves", async () => {
    const mockSocket = {
      on: jest.fn(),
      connected: true,
      emit: jest.fn()
    }
    const synchronizer = new Synchronizer(mockSocket, 0, "test", "fakeId")

    testJsonAutosave.task.config.autosave = false
    setupTestStoreWithMiddleware(testJsonAutosave, synchronizer)

    // Only need to test save button for manual saving
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Provider store={Session.store}>
          <TitleBar />
        </Provider>
      </ThemeProvider>
    )
    const dispatchSpy = jest.spyOn(Session, "dispatch")
    dispatchSpy.mockClear()

    const submitButton = getByTestId("Submit")
    fireEvent.click(submitButton)

    // Check that submit action was dispatched
    checkSubmitDispatch(dispatchSpy)

    // Make sure that the submit action was saved
    expect(mockSocket.emit).toHaveBeenCalled()
    const emitCalls = mockSocket.emit.mock.calls

    // 1 call: initSession and submit actions are saved together
    expect(emitCalls.length).toBe(1)

    // Each call has 2 args: event name and action message
    const emitArgs = emitCalls[0]
    expect(emitArgs.length).toBe(2)
    expect(emitArgs[0]).toBe(EventName.ACTION_SEND)
    const actionMessage: SyncActionMessageType = emitArgs[1]
    const actionPacket = actionMessage.actions
    expect(actionPacket.actions.length).toBe(2)
    // The first action should be INIT_SESSION, so the second is submit
    expect(actionPacket.actions[1].type).toBe(SUBMIT)
  })
})

/**
 * Checks that submit action was dispatched
 *
 * @param dispatchSpy
 */
function checkSubmitDispatch(
  dispatchSpy: jest.SpyInstance<
    void,
    [ActionType | ThunkAction<void, ReduxState, void, ActionType>]
  >
): void {
  expect(dispatchSpy).toHaveBeenCalled()
  const dispatchAction = dispatchSpy.mock.calls[0][0] as ActionType
  // Check type, instead of HaveBeenCalledWith, because userId may change
  expect(dispatchAction.type).toBe(SUBMIT)
}
