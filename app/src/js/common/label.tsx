import ListItem from '@material-ui/core/ListItem/ListItem'
import React from 'react'
import { makeButton } from '../components/general_button'
import { Category } from '../components/toolbar_category'
import { ListButton } from '../components/toolbar_list_button'
import { SwitchBtn } from '../components/toolbar_switch'
import { AttributeToolType } from './types'

/**
 * This is renderTemplate function that renders the category.
 * @param {string} toolType
 * @param {function} handleToggle
 * @param {string} name
 * @param {string[]} options
 */
export function renderTemplate (
  toolType: string,
  handleToggle: (switchName: string) => void,
  handleAttributeToggle: (
    toggleName: string,
    alignment: string) => void,
  getAlignmentIndex: (switName: string) => number,
  name: string,
  options: string[]
) {
  if (toolType === AttributeToolType.SWITCH) {
    return <SwitchBtn onChange={handleToggle}
    name={name} getAlignmentIndex={getAlignmentIndex} />
  } else if (toolType === AttributeToolType.LIST) {
    return (
      <ListItem dense={true} style={{ textAlign: 'center' }}>
        <ListButton
          name={name}
          values={options}
          handleAttributeToggle={handleAttributeToggle}
          getAlignmentIndex = {getAlignmentIndex}
        />
      </ListItem>
    )
  } else if (toolType === AttributeToolType.LONG_LIST) {
    return (
      <ListItem dense={true} style={{ textAlign: 'center' }}>
        <Category categories={options} headerText={name} />
      </ListItem>
    )
  }
}

/**
 * This is renderButtons function that renders the buttons in the toolbar.
 * @param {string} itemType
 * @param {string} labelType
 */
export function renderButtons (itemType: string, labelType: string) {
  if (itemType === 'video') {
    return (
      <div>
        <div>
          {makeButton('End Object Track')}
        </div>
        <div>
          {makeButton('Track-Link')}
        </div>
      </div>
    )
  }
  if (labelType === 'box2d') {
    // do nothing
  } else if (labelType === 'polygon2d' || labelType === 'lane') {
    if (labelType === 'polygon2d') {
      if (itemType === 'image') {
        return (
          <div>
            <div>
              {makeButton('Link')}
            </div>
            <div>
              {makeButton('Quick-draw')}
            </div>
          </div>
        )
      }
    } else if (labelType === 'lane') {
      // do nothing
    }
  }
}
