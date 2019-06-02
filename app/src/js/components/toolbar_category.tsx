import React from 'react';
import {categoryStyle} from '../styles/label';
import {withStyles} from '@material-ui/core/styles';
import FormControl from '@material-ui/core/FormControl';
import {ListItemText} from '@material-ui/core';
import Radio from '@material-ui/core/Radio/Radio';
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel';

interface Props {
    categories: any[];
    classes: any;
}

/**
 * This is a multipleSelect component that displays
 * all the categories as a list.
 */
class MultipleSelect extends React.Component<Props> {
    public state = {
        selectedValue: 'a'
    };

    public handleChange = (event: { target: { value: any; }; }) => {
        this.setState({selectedValue: event.target.value});
    };

    /**
     * MultipleSelect render function
     * @return {jsx} component
     */
    public render() {
        const {categories} = this.props;
        const {classes} = this.props;

        return (
            <div className={classes.root}>
                <FormControl className={classes.formControl}>
                    <ListItemText classes={{primary: classes.primary}}
                                  primary={'Label Category'}/>
                    <div className={classes.root}>
                        {categories.map((name) => (
                            <FormControlLabel
                                control={<Radio
                                    checked={this.state.selectedValue === name}
                                    onChange={this.handleChange}
                                    key={name}
                                    value={name}
                                    classes={{
                                        root: classes.checkbox,
                                        checked: classes.checked
                                    }}
                                />}
                                label={name}
                            />
                        ))}
                    </div>
                </FormControl>
            </div>
        );
    }
}

export const Category =
    withStyles(categoryStyle, {withTheme: true})(MultipleSelect);
