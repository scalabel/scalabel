import { cleanup, fireEvent, render } from '@testing-library/react'
import * as React from 'react'
import { goToItem } from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import Synchronizer from '../../js/common/synchronizer'
import { Key } from '../../js/common/types'
import Window from '../../js/components/window'
import { testJson } from '../test_image_objects'

afterEach(cleanup)

test('Item change with arrow keys', () => {
  Session.devMode = false
  initStore(testJson)
  const synchronizer = new Synchronizer(0, '', () => { return })
  render(<Window synchronizer={synchronizer}></Window>)

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
  for (let i = 1; i < testJson.task.items.length; i++) {
    fireEvent.keyDown(document, { key: Key.ARROW_RIGHT })
    state = Session.getState()
    expect(state.user.select.item).toEqual(i)
  }

  // Test that item does not go beyond bounds
  fireEvent.keyDown(document, { key: Key.ARROW_RIGHT })
  state = Session.getState()
  expect(state.user.select.item).toEqual(testJson.task.items.length - 1)

  // Test move to front
  for (let i = testJson.task.items.length - 2; i >= 0; i--) {
    fireEvent.keyDown(document, { key: Key.ARROW_LEFT })
    state = Session.getState()
    expect(state.user.select.item).toEqual(i)
  }
})
