import * as React from 'react';
import {categoryStyle} from '../styles/label';
import {withStyles} from '@material-ui/core/styles';
import FormControl from '@material-ui/core/FormControl';
import {ListItemText} from '@material-ui/core';
import Radio from '@material-ui/core/Radio/Radio';
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel';

interface ClassType {
    /** root of the category selector */
    root: string;
    /** form control tag */
    formControl: string;
    /** primary for ListItemText */
    primary: string;
    /** checkbox class */
    checkbox: string;
    /** checked selector class */
    checked: string;
}

interface Props {
    /** categories of MultipleSelect */
    categories: string[] | null;
    /** styles of MultipleSelect */
    classes: ClassType;
}

/**
 * This is a multipleSelect component that displays
 * all the categories as a list.
 */
class MultipleSelect extends React.Component<Props> {
    /**
     * This is the state of MultipleSelect
     */
    public state = {
        selectedValue: ''
    };

    /**
     * This is the handleChange function of MultipleSelect
     * that change the set the state of MultipleSelect.
     */
    public handleChange = (event: {
        /** target to change */
        target: {
            /** value to be changed */
            value: string;
        }; }) => {
        this.setState({selectedValue: event.target.value});
    };

    /**
     * Render the category in a list
     */
    public renderCategory(categories: string[], classes: ClassType) {
        return (
            <div>
                <FormControl className={classes.formControl}>
                    <ListItemText classes={{primary: classes.primary}}
                                  primary={'Label Category'}/>
                    <div className={classes.root}>
                        {categories.map((name: string, index: number) => (
                            <FormControlLabel
                                key={index}
                                control={<Radio
                                    checked={this.state.selectedValue === name}
                                    onChange={this.handleChange}
                                    key={'kk'}
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

    /**
     * MultipleSelect render function
     */
    public render() {
        const {categories} = this.props;
        const {classes} = this.props;
        if (!categories) {
            return (null);
        } else {
            return this.renderCategory(categories, classes);
        }
    }
}

export const Category =
    withStyles(categoryStyle, {withTheme: true})(MultipleSelect);
export default Category;
