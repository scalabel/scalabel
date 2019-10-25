import { ListItemText } from '@material-ui/core'
import FormControl from '@material-ui/core/FormControl'
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel'
import Radio from '@material-ui/core/Radio/Radio'
import RadioGroup from '@material-ui/core/RadioGroup'
import { withStyles } from '@material-ui/core/styles'
import RadioButtonCheckedIcon from '@material-ui/icons/RadioButtonChecked'
import RadioButtonUncheckedIcon from '@material-ui/icons/RadioButtonUnchecked'
import * as React from 'react'
import { changeSelect } from '../action/common'
import { changeSelectedLabelsCategories } from '../action/select'
import Session from '../common/session'
import { categoryStyle } from '../styles/label'
import { Component } from './component'

interface ClassType {
  /** root of the category selector */
  root: string
  /** form control tag */
  formControl: string
  /** primary for ListItemText */
  primary: string
  /** checkbox class */
  checkbox: string
  /** checked selector class */
  checked: string
}

interface Props {
  /** categories of MultipleSelect */
  categories: string[] | null
  /** styles of MultipleSelect */
  classes: ClassType
  /** header text of MultipleSelect */
  headerText: string
}

/**
 * This is a multipleSelect component that displays
 * all the categories as a list.
 */
class MultipleSelect extends Component<Props> {

  /**
   * This is the handleChange function of MultipleSelect
   * that change the set the state of MultipleSelect.
   */
  public handleChange = (event: {
    /** target to change */
    target: {
      /** value to be changed */
      value: string;
    };
  }) => {
    const state = Session.getState()
    const categoryId = state.task.config.categories.indexOf(event.target.value)
    Session.dispatch(changeSelect({ category: categoryId }))
    Session.dispatch(changeSelectedLabelsCategories(state, [categoryId]))
  }

  /**
   * Render the category in a list
   */
  public renderCategory (
    categories: string[], classes: ClassType, headerText: string) {
    const state = Session.getState()
    const currentCategoryId = state.user.select.category
    const currentCategory = state.task.config.categories[currentCategoryId]
    return (
      <div>
        <FormControl className={classes.formControl}>
          <ListItemText classes={{ primary: classes.primary }}
            primary={headerText} />
          <RadioGroup className={classes.root}>
            {categories.map((name: string, index: number) => (
              <FormControlLabel
                key={index}
                control={<Radio
                  checked={currentCategory === name}
                  onChange={this.handleChange}
                  key={'kk'}
                  value={name}
                  icon={<RadioButtonUncheckedIcon fontSize='small' />}
                  checkedIcon={<RadioButtonCheckedIcon fontSize='small' />}
                  classes={{
                    root: classes.checkbox,
                    checked: classes.checked
                  }}
                />}
                label={name}
              />
            ))}
          </RadioGroup>
        </FormControl>
      </div>
    )
  }

  /**
   * MultipleSelect render function
   */
  public render () {
    const { categories } = this.props
    const { classes } = this.props
    const { headerText } = this.props
    if (!categories) {
      return (null)
    } else {
      return this.renderCategory(categories, classes, headerText)
    }
  }
}

export const Category =
  withStyles(categoryStyle, { withTheme: true })(MultipleSelect)
export default Category
