import { ListItemText } from '@material-ui/core'
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel'
import { cleanup, fireEvent, render } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import { create } from 'react-test-renderer'
import * as action from '../../js/action/common'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore, loadImages } from '../../js/common/session_init'
import { ToolBar } from '../../js/components/toolbar'
import { Category } from '../../js/components/toolbar_category'
import { ListButton } from '../../js/components/toolbar_list_button'
import { makeLabel } from '../../js/functional/states'
import { testJson } from '../test_image_objects'
import { testJson as testTrackJson } from '../test_track_objects'

let handleToggleWasCalled: boolean = false
const testValues = ['NA', 'A', 'B', 'C']
const selected: {[key: string]: string} = {}

/**
 * dummy attribute toggle function to test if the correct toggle action was
 * correctly called
 * @param toggleName
 * @param alignment
 * @param isTag
 */
const dummyAttributeToggle = (
  toggleName: string,
  alignment: string
): void => {
  selected[toggleName] = alignment
  handleToggleWasCalled = true
}

const dummyGetAlignmentIndex = (toggleName: string) => {
  const index = testValues.indexOf(selected[toggleName])
  return index >= 0 ? index : 0
}

beforeEach(() => {
  handleToggleWasCalled = false
})

afterEach(cleanup)

describe('Toolbar category setting', () => {
  test('Category selection', () => {
    const { getByLabelText } = render(
      <Category categories={['A', 'B']} headerText={'Label Category'} />)
    const selectedValued = getByLabelText(/A/i)
    expect(selectedValued.getAttribute('value')).toEqual('A')
    const radio = getByLabelText('A')
    fireEvent.change(radio, { target: { value: 'B' } })
    // expect state to be changed
    expect(radio.getAttribute('value')).toBe('B')
  })

  test('Test elements in Category', () => {
    const category = create(
      <Category categories={['A', 'B']} headerText={'Label Category'} />)
    const root = category.root
    expect(root.props.categories[0].toString()).toBe('A')
    expect(root.props.categories[1].toString()).toBe('B')
    expect(root.findByType(ListItemText).props.primary)
      .toBe('Label Category')
  })

  test('Category by type', () => {
    const category = create(
      <Category categories={['OnlyCategory']} headerText={'Label Category'} />)
    const root = category.root
    expect(root.findByType(FormControlLabel).props.label)
      .toBe('OnlyCategory')
  })

  test('Null category', () => {
    const category = create(
      <Category categories={null} headerText={'Label Category'} />)
    const root = category.getInstance()
    expect(root).toBe(null)
  })
})

describe('test Delete', () => {
  test('Delete by keyboard', () => {
    Session.devMode = false
    initStore(testJson)
    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    render(
    <ToolBar
      ref={toolbarRef}
      categories={null}
      attributes={[]}
      labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
      toolbarRef.current.componentDidMount()
    }
    for (let itemIndex = 0; itemIndex < 3; itemIndex += 1) {
      Session.dispatch(action.goToItem(itemIndex))
      const label = makeLabel({ item: itemIndex })
      Session.dispatch(action.addLabel(itemIndex, label))
      Session.dispatch(action.addLabel(itemIndex, label))
      const secondLabelId = Session.getState().task.status.maxLabelId
      Session.dispatch(action.addLabel(itemIndex, label))
      Session.dispatch(action.addLabel(itemIndex, label))
      fireEvent.keyDown(document, { key: 'Backspace' })
      let item = Session.getState().task.items[itemIndex]
      expect(_.size(item.labels)).toBe(3)
      expect(secondLabelId in item.labels).toBe(true)
      Session.dispatch(selectLabel(
        Session.getState().user.select.labels, itemIndex, secondLabelId))
      fireEvent.keyDown(document, { key: 'Backspace' })
      item = Session.getState().task.items[itemIndex]
      expect(_.size(item.labels)).toBe(2)
      expect(secondLabelId in item.labels).toBe(false)
    }
  })
})

describe('test functionality for attributes with multiple values', () => {
  test('proper initialization', () => {
    const { getByTestId } = render(
      <ListButton
        name={'test'}
        values={testValues}
        handleAttributeToggle={dummyAttributeToggle}
        getAlignmentIndex={dummyGetAlignmentIndex}
      />
    )
    expect(handleToggleWasCalled).toBe(false)
    const NAButton = getByTestId('toggle-button-NA') as (HTMLButtonElement)
    expect(NAButton.className).toContain('selected')
  })
  test('changing selected attribute calls callback for tag labeling', () => {
    const { getByTestId } = render(
      <ListButton
        name={'test'}
        values={testValues}
        handleAttributeToggle={dummyAttributeToggle}
        getAlignmentIndex={dummyGetAlignmentIndex}

      />
    )
    let AButton = getByTestId('toggle-button-A') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(false)
    fireEvent.click(AButton)
    AButton = getByTestId('toggle-button-A') as (HTMLButtonElement)
    let NAButton = getByTestId('toggle-button-NA') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(true)
    AButton = getByTestId('toggle-button-A') as (HTMLButtonElement)
    expect(AButton.className).toContain('selected')
    expect(NAButton.className).not.toContain('selected')
    handleToggleWasCalled = false
    fireEvent.click(NAButton)
    NAButton = getByTestId('toggle-button-NA') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(true)
    expect(NAButton.className).toContain('selected')
  })
})

describe('test track', () => {
  test('Delete by click toolbar button', () => {
    Session.devMode = false
    initStore(testTrackJson)
    loadImages()
    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText } = render(
    <ToolBar
      ref={toolbarRef}
      categories={null}
      attributes={[]}
      labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
      toolbarRef.current.componentDidMount()
    }

    Session.dispatch(action.goToItem(1))
    let state = Session.getState()
    const trackLabels = state.task.tracks[3].labels
    const lblInItm2 = trackLabels[2]
    const lblInItm3 = trackLabels[3]
    const lblInItm4 = trackLabels[4]
    const lblInItm5 = trackLabels[5]
    expect(_.size(state.task.tracks[3].labels)).toBe(6)
    expect(_.size(state.task.items[2].labels)).toBe(3)
    expect(_.size(state.task.items[2].shapes)).toBe(3)
    Session.dispatch(selectLabel(
      Session.getState().user.select.labels, 1, 70))
    fireEvent(
      getByText('End Object Tracking'),
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )
    state = Session.getState()
    expect(_.size(state.task.tracks[3].labels)).toBe(2)
    expect(state.task.items[2].labels[lblInItm2]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm3]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm4]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm5]).toBeUndefined()
  })
  test('Merge by click toolbar button', () => {
    Session.devMode = false
    initStore(testTrackJson)
    loadImages()
    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText } = render(
    <ToolBar
      ref={toolbarRef}
      categories={null}
      attributes={[]}
      labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
      toolbarRef.current.componentDidMount()
    }

    Session.dispatch(action.goToItem(3))
    let state = Session.getState()
    expect(_.size(state.task.tracks[2].labels)).toBe(4)
    expect(_.size(state.task.tracks[9].labels)).toBe(1)
    expect(state.task.items[5].labels[203].track).toEqual(9)

    Session.dispatch(selectLabel(
      Session.getState().user.select.labels, 3, 49))
    fireEvent(
      getByText('Track-Link'),
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )

    Session.dispatch(action.goToItem(5))
    Session.dispatch(selectLabel(
      Session.getState().user.select.labels, 5, 203))
    fireEvent(
      getByText('Finish Track-Link'),
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )

    state = Session.getState()
    expect(_.size(state.task.tracks[2].labels)).toBe(5)
    expect(state.task.tracks[9]).toBeUndefined()
    expect(state.task.items[5].labels[203].track).toEqual(2)
  })
})
