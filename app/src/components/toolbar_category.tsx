import { ListItemText, ListItem } from "@material-ui/core"
import FormControl from "@material-ui/core/FormControl"
import { withStyles } from "@material-ui/core/styles"
import ToggleButton from "@material-ui/lab/ToggleButton"
import ToggleButtonGroup from "@material-ui/lab/ToggleButtonGroup"
// import Typography from "@material-ui/core/Typography"
import * as React from "react"

import { changeSelect, makeSequential } from "../action/common"
import { changeSelectedLabelsCategories } from "../action/select"
import { dispatch, getState } from "../common/session"
import { categoryStyle } from "../styles/label"
import { BaseAction } from "../types/action"
import { Component } from "./component"

/**
 * This is the handleChange function of MultipleSelect
 * that change the set the state of MultipleSelect.
 *
 * @param _event
 * @param categoryIndex
 */
function handleChange(
  _event: React.MouseEvent<HTMLElement>,
  categoryIndex: number | null
): void {
  const state = getState()
  if (categoryIndex !== null) {
    const actions: BaseAction[] = []
    actions.push(changeSelect({ category: categoryIndex }))
    // Update categories if any labels are selected
    if (Object.keys(state.user.select.labels).length > 0) {
      actions.push(changeSelectedLabelsCategories(state, [categoryIndex]))
    }
    dispatch(makeSequential(actions))
  }
}

interface ClassType {
  /** root of the category selector */
  root: string
  /** form control tag */
  formControl: string
  /** primary for ListItemText */
  primary: string
  /** button style */
  button: string
  /** button group style */
  buttonGroup: string
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
   * Render the category in a list
   *
   * @param categories
   * @param classes
   * @param headerText
   */
  public renderCategory(
    categories: string[],
    classes: ClassType,
    headerText: string
  ): JSX.Element {
    return (
      <>
        <FormControl className={classes.formControl}>
          <ListItem dense={true} className={classes.primary}>
            <ListItemText
              classes={{ primary: classes.primary }}
              primary={headerText}
            />
          </ListItem>
          <ToggleButtonGroup
            className={classes.buttonGroup}
            orientation="vertical"
            exclusive
            onChange={handleChange}
            value={getState().user.select.category}
            aria-label="vertical outlined primary button group"
          >
            {categories.map((name: string, index: number) => (
              <ToggleButton
                className={classes.button}
                key={`category-${name}`}
                value={index}
                disableRipple={true}
              >
                {name}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </FormControl>
      </>
    )
  }

  /**
   * MultipleSelect render function
   */
  public render(): React.ReactNode {
    const { categories } = this.props
    const { classes } = this.props
    const { headerText } = this.props
    if (categories === null) {
      return null
    } else {
      return this.renderCategory(categories, classes, headerText)
    }
  }
}

export const Category = withStyles(categoryStyle, { withTheme: true })(
  MultipleSelect
)
export default Category
