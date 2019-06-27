import React from 'react';
import { ListItem, ListItemText } from '@material-ui/core';
import { switchStyle } from '../styles/label';
import Switch from '@material-ui/core/Switch';
import { withStyles } from '@material-ui/core/styles';
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction';

/**
 * Interface used for props.
 */
interface Props {
  /** onChange function */
  onChange: any;
  /** values passed to onChange function . */
  value: any;
  /** styles of SwitchButton. */
  classes: any;
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
  public render() {
    const { onChange, value, classes } = this.props;

    // @ts-ignore
    return (
      <ListItem>
        <ListItemText classes={{ primary: classes.primary }}
          primary={value} />
        <ListItemSecondaryAction>
          <Switch
            classes={{
              switchBase: classes.colorSwitchBase,
              checked: classes.colorChecked
            }}
            onChange={onChange(value)}
            color='default'
          />
        </ListItemSecondaryAction>
      </ListItem>
    );
  }
}

export const SwitchBtn = withStyles(switchStyle)(SwitchButton);
