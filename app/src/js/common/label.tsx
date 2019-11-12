import ListItem from '@material-ui/core/ListItem/ListItem'
import React from 'react'
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
