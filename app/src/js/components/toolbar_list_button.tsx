import React from 'react';
import {Validator} from 'prop-types';
import {withStyles} from '@material-ui/core/styles';
import {toggleButtonStyle, listButtonStyle} from '../styles/label';
import ToggleButton from '@material-ui/lab/ToggleButton';
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup';
import ListItem from '@material-ui/core/ListItem';
import List from '@material-ui/core/List';
import {ListItemText} from '@material-ui/core';

interface Props {
    classes: any;
    name: any;
    values: string[];
}

class ToggleButtons extends React.Component<Props> {
    public state = {
        alignment: this.props.values[0]
    };
    public handleAlignment =
        (event: any, alignment: any) => this.setState({alignment});
    public static propTypes: { classes: Validator<NonNullable<object>> };

    public render() {
        const {name, classes, values} = this.props;
        const {alignment} = this.state;
        const ToggleBtn = withStyles(toggleButtonStyle)(ToggleButton);

        return (
            <List style={{width: '100%'}}>
                <ListItemText style={{textAlign: 'center', width: '100%'}}
                              classes={{primary: classes.primary}}
                              primary={name}/>
                <ListItem style={{width: '100%'}}>
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
        );
    }
}

export const ListButton = withStyles(listButtonStyle)(ToggleButtons);
