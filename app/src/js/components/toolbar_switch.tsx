import { ListItem, ListItemText } from '@material-ui/core'
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction'
import { withStyles } from '@material-ui/core/styles'
import Switch from '@material-ui/core/Switch'
import React from 'react'
import { switchStyle } from '../styles/label'

interface ClassType {
  /** root class */
  root: string,
  /** primary text class */
  primary: string,
  /** switch color class */
  switchBase: string
  /** checked state class */
  checked: string
  /** track class */
  track: string
}

/**
 * Interface used for props.
 */
interface Props {
  /** onChange function */
  onChange: (switchName: string) => () => void
  /** values passed to onChange function . */
  value: string
  /** styles of SwitchButton. */
  classes: ClassType
}

/**
 * This is a Switch Button component that
 * displays the list of selections.
 * @param {object} Props
 */
class SwitchButton extends React.Component<Props> {
  /**
   * SwitchButton render function
   */
  public render () {
    const { onChange, value, classes } = this.props

    return (
      <ListItem dense={true}>
        <ListItemText classes={{ primary: classes.primary }}
          primary={value} />
        <ListItemSecondaryAction>
          <Switch
            classes={{
              switchBase: classes.switchBase,
              checked: classes.checked,
              track: classes.track
            }}
            onChange={onChange(value)}
          />
        </ListItemSecondaryAction>
      </ListItem>
    )
  }
}

export const SwitchBtn = withStyles(switchStyle)(SwitchButton)
