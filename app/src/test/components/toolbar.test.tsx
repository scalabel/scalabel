import { ListItemText } from '@material-ui/core'
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel'
import { cleanup, fireEvent, render } from '@testing-library/react'
import _ from 'lodash'
import * as React from 'react'
import { create } from 'react-test-renderer'
import * as action from '../../js/action/common'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { ToolBar } from '../../js/components/toolbar'
import { Category } from '../../js/components/toolbar_category'
import { ListButton } from '../../js/components/toolbar_list_button'
import { makeLabel } from '../../js/functional/states'
import { testJson } from '../test_image_objects'

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
    render(
    <ToolBar
      categories={null}
      attributes={[]}
      itemType={'itemType'}
      labelType={'labelType'}
      />
    )
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
      Session.dispatch(selectLabel(secondLabelId))
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
