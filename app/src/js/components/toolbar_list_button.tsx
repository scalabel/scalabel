import { ListItemText } from '@material-ui/core'
import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import { withStyles } from '@material-ui/core/styles'
import ToggleButton from '@material-ui/lab/ToggleButton'
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup'
import { Validator } from 'prop-types'
import React from 'react'
import { listButtonStyle, toggleButtonStyle } from '../styles/label'

interface Props {
  /** styles of ToggleButtons */
  classes: any
  /** name of ToggleButtons */
  name: any
  /** values of ToggleButtons */
  values: string[]
}

/**
 * This is ToggleButtons component that displays
 * the everything post in the dashboard.
 * @param {object} props
 */
class ToggleButtons extends React.Component<Props> {
  /** propTypes of ToggleButtons */
  public static propTypes: {
    /** type of classes */
    classes: Validator<NonNullable<object>>
  }
  /** state of ToggleButtons */
  public state = {
    alignment: this.props.values[0]
  }
  /** handleAlignment of ToggleButtons that align buttons */
  public handleAlignment =
    (_event: any, alignment: any) => this.setState({ alignment })
  /** render function of ToggleButtons */
  public render () {
    const { name, classes, values } = this.props
    const { alignment } = this.state
    const ToggleBtn = withStyles(toggleButtonStyle)(ToggleButton)

    return (
      <List style={{ width: '100%' }}>
        <ListItemText style={{ textAlign: 'center', width: '100%' }}
          classes={{ primary: classes.primary }}
          primary={name} />
        <ListItem style={{ width: '100%' }}>
          <div className={classes.toggleContainer}
            style={{
              marginRight: 'auto',
              marginLeft: 'auto'
            }}
          >
            <ToggleButtonGroup
              className={classes.buttonGroup}
              value={alignment}
              exclusive
              onChange={this.handleAlignment}
            >
              {values.map((element: any) => (
                <ToggleBtn
                  value={element}> {element} </ToggleBtn>
              ))}
            </ToggleButtonGroup>
          </div>
        </ListItem>
      </List>
    )
  }
}

export const ListButton = withStyles(listButtonStyle)(ToggleButtons)
