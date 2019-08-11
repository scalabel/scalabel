import { IconButton } from '@material-ui/core'
import { cleanup } from '@testing-library/react'
import * as React from 'react'
import { create } from 'react-test-renderer'
import * as types from '../../js/action/types'
import Session, { ConnectionStatus } from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import TitleBar from '../../js/components/title_bar'
import { autoSaveTestJson, dummyNewLabel } from '../test_objects'

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
  beforeEach(() => {
    initStore({ testKey: 'testValue' })
    jest.clearAllMocks()
  })

  // titlebar contains save button
  const titleBar = create(
    <TitleBar
      title={'title'}
      instructionLink={'instructionLink'}
      dashboardLink={'dashboardLink'}
    />
  )
  const buttonContainer = titleBar.root.findByProps({ title: 'Save' })
  const saveButton = buttonContainer.findByType(IconButton)

  test('Save button triggers save', () => {
    saveButton.props.onClick()
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalledWith(
      expect.stringContaining('"testKey":"testValue"')
    )
    xhrMockClass.onreadystatechange()
  })

  test('Sync status is correct during save', () => {
    expect(Session.status).toBe(ConnectionStatus.SAVED)
    saveButton.props.onClick()
    expect(Session.status).toBe(ConnectionStatus.SAVING)
    xhrMockClass.onreadystatechange()
    expect(Session.status).toBe(ConnectionStatus.SAVED)
  })

  test('Alerts on save failure', () => {
    xhrMockClass.response = JSON.stringify(null)

    saveButton.props.onClick()
    xhrMockClass.onreadystatechange()
    expect(windowInterface.alert).toBeCalled()
    expect(Session.status).toBe(ConnectionStatus.SAVED)

    xhrMockClass.response = JSON.stringify(0)
  })
})

// re-enable once autosave is implemented
xdescribe('Autosave', () => {
  beforeAll(() => {
    initStore(autoSaveTestJson)

    // titlebar contains autosave subscriber
    create(
      <TitleBar
        title={'title'}
        instructionLink={'instructionLink'}
        dashboardLink={'dashboardLink'}
      />
    )
    // dispatch an action to trigger/init autosave subscriber
    Session.dispatch({
      type: types.IMAGE_ZOOM, ratio: 1.05,
      viewOffsetX: 0, viewOffsetY: 0, sessionId: Session.id
    })
    xhrMockClass.onreadystatechange()
  })

  test('No save if data does not change', () => {
    Session.dispatch({
      type: types.IMAGE_ZOOM, ratio: 1.05,
      viewOffsetX: 0, viewOffsetY: 0, sessionId: Session.id
    })
    expect(xhrMockClass.open).not.toBeCalled()
    expect(xhrMockClass.send).not.toBeCalled()
  })

  test('Saves if label is added', () => {
    Session.dispatch({
      type: types.ADD_LABEL,
      itemIndex: 0,
      sessionId: Session.id,
      label: dummyNewLabel,
      shapes: [
        { x1: 1, y1: 1, x2: 2, y2: 2 }
      ]
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })

  test('Saves if label shape is changed', () => {
    Session.dispatch({
      type: types.CHANGE_LABEL_SHAPE,
      sessionId: Session.id,
      itemIndex: 0,
      shapeId: 0,
      props: { x1: 0, y1: 0, x2: 1, y2: 2 }
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })

  test('Saves if label categories change', () => {
    Session.dispatch({
      type: types.CHANGE_LABEL_PROPS,
      sessionId: Session.id,
      itemIndex: 0,
      labelId: 0,
      props: {
        category: [1]
      }
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })

  test('Saves if label attributes change', () => {
    Session.dispatch({
      type: types.CHANGE_LABEL_PROPS,
      sessionId: Session.id,
      itemIndex: 0,
      labelId: 0,
      props: {
        attributes: {
          0: [0],
          1: [1]
        }
      }
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })

  test('Saves if label is deleted', () => {
    Session.dispatch({
      type: types.DELETE_LABEL,
      sessionId: Session.id,
      itemIndex: 0,
      labelId: 2
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })

  test('Saves if user goes to next item', () => {
    Session.dispatch({
      type: types.GO_TO_ITEM,
      sessionId: Session.id,
      itemIndex: 1
    })
    expect(xhrMockClass.open).toBeCalled()
    expect(xhrMockClass.send).toBeCalled()
    xhrMockClass.onreadystatechange()
  })
})
