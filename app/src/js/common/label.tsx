import ListItem from '@material-ui/core/ListItem/ListItem'
import React from 'react'
import { genButton } from '../components/general_button'
import { ListButton } from '../components/toolbar_list_button'
import { SwitchBtn } from '../components/toolbar_switch'

/**
 * This is renderTemplate function that renders the category.
 * @param {string} toolType
 * @param {function} handleToggle
 * @param {string} name
 * @param {string[]} values
 */
export function renderTemplate (
  toolType: string, handleToggle: (switchName: string) => () => void,
  name: string, values: string[]) {
  if (toolType === 'switch') {
    return (
      <SwitchBtn onChange={handleToggle} value={name} />
    )
  } else if (toolType === 'list') {
    return (
      <ListItem style={{ textAlign: 'center' }} >
        <ListButton name={name} values={values} />
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
          {genButton({ name: 'End Object Track' })}
        </div>
        <div>
          {genButton({ name: 'Track-Link' })}
        </div>
      </div>
    )
  }
  if (labelType === 'box2d') {
    // do nothing
  } else if (labelType === 'segmentation' || labelType === 'lane') {
    if (labelType === 'segmentation') {
      if (itemType === 'image') {
        return (
          <div>
            <div>
              {genButton({ name: 'Link' })}
            </div>
            <div>
              {genButton({ name: 'Quick-draw' })}
            </div>
          </div>
        )
      }
    } else if (labelType === 'lane') {
      // do nothing
    }
  }
}
