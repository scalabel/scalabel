import React from 'react';
import {ListItem, ListItemText} from '@material-ui/core';
import {switchStyle} from '../styles/label';
import Switch from '@material-ui/core/Switch';
import {withStyles} from '@material-ui/core/styles';
import ListItemSecondaryAction from '@material-ui/core/ListItemSecondaryAction';

interface Props {
    onChange: any;
    value: any;
    classes: any;
}

/**
 * This is a Switch Button component that
 * displays the list of selections.
 * @param {object} Props
 * @return {jsx} component
 */
class SwitchButton extends React.Component<Props> {
    /**
     * SwitchButton render function
     * @return {jsx} component
     */
    public render() {
        const {onChange, value, classes} = this.props;

        return (
            <ListItem>
                <ListItemText classes={{primary: classes.primary}}
                              primary={value}/>
                <ListItemSecondaryAction>
                    <Switch
                        classes={{
                            switchBase: classes.colorSwitchBase,
                            checked: classes.colorChecked,
                            bar: classes.colorBar
                        }}
                        onChange={onChange(value)}
                    />
                </ListItemSecondaryAction>
            </ListItem>
        );
    }
}

export const SwitchBtn = withStyles(switchStyle)(SwitchButton);
