import { ListItem, ListItemText } from '@material-ui/core'
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction'
import { withStyles } from '@material-ui/core/styles'
import Switch from '@material-ui/core/Switch'
import React from 'react'
import { switchStyle } from '../styles/label'
import { Component } from './component'

interface ClassType {
  /** root class */
  root: string,
  /** primary text class */
  primary: string,
  /** switch color class */
  switchBase: string
  /** track class */
  track: string
}

/**
 * Interface used for props.
 */
interface Props {
  /** onChange function */
  onChange: (switchName: string) => void
  /** name of the switch */
  name: string
  /** value of the switch */
  value: number
  /** styles of SwitchButton. */
  classes: ClassType
}

/**
 * This is a Switch Button component that
 * displays the list of selections.
 * @param {object} Props
 */
class SwitchButton extends Component<Props> {
  /**
   * SwitchButton render function
   */
  public render () {
    const { onChange, name, value, classes } = this.props

    return (
      <ListItem dense={true}>
        <ListItemText classes={{ primary: classes.primary }}
          primary={name} />
        <ListItemSecondaryAction>
          <Switch
            classes={{
              switchBase: classes.switchBase,
              track: classes.track
            }}
            checked={value > 0}
            onChange={() => onChange(name)}
          />
        </ListItemSecondaryAction>
      </ListItem>
    )
  }
}

export const SwitchBtn = withStyles(switchStyle)(SwitchButton)
