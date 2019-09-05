import { cleanup, fireEvent, render } from '@testing-library/react'
import * as React from 'react'
import { ListButton } from '../../js/components/toolbar_list_button'

let handleToggleWasCalled: boolean = false
const testValues = ['NA', 'A', 'B', 'C']
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
  toggleName = toggleName
  alignment = alignment
  handleToggleWasCalled = true
}
beforeEach(() => {
  handleToggleWasCalled = false
})
afterEach(cleanup)
// TODO: update these tests when attributes is added for box2d
describe('test functionality for attributes with multiple values', () => {
  test('proper initialization', () => {
    const { getByTestId } = render(
      <ListButton
        name={'test'}
        values={testValues}
        handleAttributeToggle={dummyAttributeToggle}
        initialAlignmentIndex={0}
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
        initialAlignmentIndex={0}
      />
    )
    let AButton = getByTestId('toggle-button-A') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(false)
    fireEvent.click(AButton)
    AButton = getByTestId('toggle-button-A') as (HTMLButtonElement)
    let NAButton = getByTestId('toggle-button-NA') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(true)
    expect(AButton.className).toContain('selected')
    expect(NAButton.className).not.toContain('selected')
    handleToggleWasCalled = false
    fireEvent.click(NAButton)
    NAButton = getByTestId('toggle-button-NA') as (HTMLButtonElement)
    expect(handleToggleWasCalled).toBe(true)
    expect(NAButton.className).toContain('selected')
  })
})
