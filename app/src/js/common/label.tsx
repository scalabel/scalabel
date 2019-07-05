import ListItem from '@material-ui/core/ListItem/ListItem'
import React from 'react'
import { genButton } from '../components/general_button'
import { ListButton } from '../components/toolbar_list_button'
import { SwitchBtn } from '../components/toolbar_switch'

/**
 * This is renderTemplate function that renders the category.
 * @param {Object} toolType
 * @param {Object} handleToggle
 * @param {Object} name
 * @param {Object} values
 */
export function renderTemplate (toolType: any, handleToggle: any,
                                name: any, values: any) {
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
 * @param {Object} itemType
 * @param {Object} labelType
 */
export function renderButtons (itemType: any, labelType: any) {
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
