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
  /** gets alignment (either 0 or 1) */
  getAlignmentIndex?: (switchName: string) => number
  /** optional value, overrides get alignment index */
  value?: number
  /** name of the switch */
  name: string
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
    const { onChange, name, value, getAlignmentIndex, classes } = this.props

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
            checked={(value === undefined) ?
              (getAlignmentIndex === undefined) ?
              true : (getAlignmentIndex(name) > 0)
              : (value > 0)}
            onChange={() => onChange(name)}
          />
        </ListItemSecondaryAction>
      </ListItem>
    )
  }
}

export const SwitchBtn = withStyles(switchStyle)(SwitchButton)
