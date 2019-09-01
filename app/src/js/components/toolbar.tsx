import Divider from '@material-ui/core/Divider'
import List from '@material-ui/core/List/List'
import ListItem from '@material-ui/core/ListItem'
import React from 'react'
import { deleteLabel } from '../action/common'
import { renderButtons, renderTemplate } from '../common/label'
import Session from '../common/session'
import { Attribute } from '../functional/types'
import { makeButton } from './general_button'
import { Category } from './toolbar_category'

/**
 * callback function for delete label button
 */
function onDeleteLabel () {
  const select = Session.getState().user.select
  if (select.label >= 0) {
    Session.dispatch(deleteLabel(select.item, select.label))
  }
}

/** This is the interface of props passed to ToolBar */
interface Props {
  /** categories of ToolBar */
  categories: string[] | null
  /** attributes of ToolBar */
  attributes: Attribute[]
  /** itemType of ToolBar 'video' | 'image' */
  itemType: string
  /** labelType of ToolBar 'box2d' | 'segmentation' | 'lane' */
  labelType: string
}
/**
 * This is ToolBar component that displays
 * all the attributes and categories for the 2D bounding box labeling tool
 */
export class ToolBar extends React.Component<Props> {
  constructor (props: Readonly<Props>) {
    super(props)
    this.handleToggle = this.handleToggle.bind(this)
    this.state = {
      checked: []
    }
  }
  /**
   * ToolBar render function
   * @return component
   */
  public render () {
    const { categories, attributes, itemType, labelType } = this.props

    return (
      <div>
        <ListItem style={{ textAlign: 'center' }} >
          <Category categories={categories} headerText={'Label Category'}/>
        </ListItem>
        <Divider variant='middle' />
        <List>
          {attributes.map((element: Attribute) => (
            renderTemplate(element.toolType, this.handleToggle,
              element.name, element.values)
          ))}
        </List>
        <div>
          <div>
            {makeButton('Delete', onDeleteLabel)}
          </div>
          {renderButtons(itemType, labelType)}
        </div>
      </div>
    )
  }
  /**
   * This function updates the checked list of switch buttons.
   * @param {string} switchName
   */
  private readonly handleToggle = (switchName: string) => () => {
    // @ts-ignore
    const { checked } = this.state
    const currentIndex = checked.indexOf(switchName)
    const newChecked = [...checked]

    if (currentIndex === -1) {
      newChecked.push(switchName)
    } else {
      newChecked.splice(currentIndex, 1)
    }

    this.setState({
      checked: newChecked
    })
  }
}
