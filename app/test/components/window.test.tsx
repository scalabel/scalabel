import { cleanup, fireEvent, render } from "@testing-library/react"
import * as React from "react"
import { Provider } from "react-redux"

import { addViewerConfig, goToItem } from "../../src/action/common"
import Session from "../../src/common/session"
import Window from "../../src/components/window"
import { Key, ViewerConfigTypeName } from "../../src/const/common"
import { makeDefaultViewerConfig } from "../../src/functional/states"
import { State } from "../../src/types/state"
import { testJson } from "../test_states/test_image_objects"
import { setupTestStore } from "./util"

afterEach(cleanup)

test("Item change with arrow keys", () => {
  setupTestStore(testJson)
  const numItems = (testJson as State).task.items.length

  const config = makeDefaultViewerConfig(ViewerConfigTypeName.IMAGE, 0, -1)
  expect(config).not.toBeNull()
  if (config !== null) {
    Session.dispatch(addViewerConfig(0, config))
  }
  render(
    <Provider store={Session.store}>
      <Window />
    </Provider>
  )

  Session.dispatch(goToItem(0))

  let state = Session.getState()
  expect(state.user.select.item).toEqual(0)

  // Test that item is not changed to negative
  fireEvent.keyDown(document, { key: Key.ARROW_LEFT })
  state = Session.getState()
  expect(state.user.select.item).toEqual(0)

  // Test forward/back
  fireEvent.keyDown(document, { key: Key.ARROW_RIGHT })
  state = Session.getState()
  expect(state.user.select.item).toEqual(1)
  fireEvent.keyDown(document, { key: Key.ARROW_LEFT })
  state = Session.getState()
  expect(state.user.select.item).toEqual(0)

  // Test move to end
  for (let i = 1; i < numItems; i++) {
    fireEvent.keyDown(document, { key: Key.ARROW_RIGHT })
    state = Session.getState()
    expect(state.user.select.item).toEqual(i)
  }

  // Test that item does not go beyond bounds
  fireEvent.keyDown(document, { key: Key.ARROW_RIGHT })
  state = Session.getState()
  expect(state.user.select.item).toEqual(numItems - 1)

  // Test move to front
  for (let i = numItems - 2; i >= 0; i--) {
    fireEvent.keyDown(document, { key: Key.ARROW_LEFT })
    state = Session.getState()
    expect(state.user.select.item).toEqual(i)
  }
})
