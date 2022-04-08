import { ListItemText, ListItem } from "@material-ui/core"
import FormControl from "@material-ui/core/FormControl"
import { withStyles } from "@material-ui/core/styles"
import ToggleButton from "@material-ui/lab/ToggleButton"
import ToggleButtonGroup from "@material-ui/lab/ToggleButtonGroup"
import TreeView from "@material-ui/lab/TreeView"
import TreeItem from "@material-ui/lab/TreeItem"
import ExpandMoreIcon from "@material-ui/icons/ExpandMore"
import ChevronRightIcon from "@material-ui/icons/ChevronRight"
import * as React from "react"

import { changeSelect, makeSequential } from "../action/common"
import { changeSelectedLabelsCategories } from "../action/select"
import { dispatch, getState } from "../common/session"
import { categoryStyle } from "../styles/label"
import { BaseAction } from "../types/action"
import { Component } from "./component"
import { Category } from "../types/state"
import { LabelTypeName } from "../const/common"

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
    // Update categories if any non-plane labels are selected
    const hasSelectedLabels = Object.keys(state.user.select.labels).length > 0
    if (hasSelectedLabels) {
      const labelIds = Object.values(state.user.select.labels)
      const groundPlaneSelected =
        state.task.items[state.user.select.item].labels[labelIds[0][0]].type ===
        LabelTypeName.PLANE_3D
      if (!groundPlaneSelected) {
        actions.push(changeSelectedLabelsCategories(state, [categoryIndex]))
      }
    }
    dispatch(makeSequential(actions))
  }
}

/**
 * This is the nodeSelect function of TreeView
 * that change the set the state of TreeView.
 *
 * @param _event
 * @param categoryIndex
 */
function handleTreeSelect(
  _event: React.ChangeEvent<{}>,
  categoryIndex: string[]
): void {
  if (categoryIndex.includes("NotLeaf")) {
    return
  }
  const state = getState()
  const actions: BaseAction[] = []
  actions.push(changeSelect({ category: Number(categoryIndex) }))

  // Update categories if any non-plane labels are selected
  const hasSelectedLabels = Object.keys(state.user.select.labels).length > 0
  if (hasSelectedLabels) {
    const labelIds = Object.values(state.user.select.labels)
    const groundPlaneSelected =
      state.task.items[state.user.select.item].labels[labelIds[0][0]].type ===
      LabelTypeName.PLANE_3D
    if (!groundPlaneSelected) {
      actions.push(
        changeSelectedLabelsCategories(state, [Number(categoryIndex)])
      )
    }
  }
  dispatch(makeSequential(actions))
}

/**
 * Create a map for quick lookup of category data
 *
 * @param categories the categories from config file
 * returns a map from category value to its index
 */
function getCategoryMap(categories: string[]): { [key: string]: number } {
  const categoryNameMap: { [key: string]: number } = {}
  for (let catInd = 0; catInd < categories.length; catInd++) {
    // Map category names to their indices
    const category = categories[catInd]
    categoryNameMap[category] = catInd
  }
  return categoryNameMap
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
  /** tree view style */
  treeView: string
  /** tree item root style */
  treeItemRoot: string
  /** tree item iconcontainer style */
  treeItemIconContainer: string
  /** tree item content style */
  treeItemContent: string
  /** tree item group style */
  treeItemGroup: string
  /** tree item label style */
  treeItemLabel: string
  /** tree item label text style */
  treeItemLabelText: string
  /** tree item first-level outline */
  treeItemOutline: string
}

interface Props {
  /** categories of MultipleSelect */
  categories: string[] | null
  /** tree categories of MultipleSelect */
  treeCategories: Category[] | null
  /** styles of MultipleSelect */
  classes: ClassType
  /** header text of MultipleSelect */
  headerText: string
}

/**
 * Function to create a tree-level category select menu
 *
 * @param treeCategory
 * @param categoryNameMap
 * @param treeLevel
 * @param classes
 */
function renderTreeCategory(
  treeCategory: Category,
  categoryNameMap: { [key: string]: number },
  treeLevel: number,
  classes: ClassType
): JSX.Element {
  const isLeaf: boolean = !Array.isArray(treeCategory.subcategories)
  const nodeId = isLeaf
    ? categoryNameMap[treeCategory.name].toString()
    : treeCategory.name + "-" + treeLevel.toString() + "-NotLeaf"
  return (
    <TreeItem
      key={treeCategory.name}
      nodeId={nodeId}
      label={
        <div className={classes.treeItemLabelText}>{treeCategory.name}</div>
      }
      classes={{
        root: classes.treeItemRoot,
        iconContainer: classes.treeItemIconContainer,
        content: classes.treeItemContent,
        group: classes.treeItemGroup,
        label: classes.treeItemLabel
      }}
      className={treeLevel === 0 ? classes.treeItemOutline : undefined}
    >
      {Array.isArray(treeCategory.subcategories)
        ? treeCategory.subcategories.map((category) =>
            renderTreeCategory(
              category,
              categoryNameMap,
              treeLevel + 1,
              classes
            )
          )
        : null}
    </TreeItem>
  )
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
   * @param treeCategories
   * @param classes
   * @param headerText
   */
  public renderCategory(
    categories: string[],
    treeCategories: Category[] | null,
    classes: ClassType,
    headerText: string
  ): JSX.Element {
    const categoryNameMap = getCategoryMap(categories)
    return (
      <>
        <FormControl className={classes.formControl}>
          <ListItem dense={true} className={classes.primary}>
            <ListItemText
              classes={{ primary: classes.primary }}
              primary={headerText}
            />
          </ListItem>
          {treeCategories !== null ? (
            <TreeView
              onNodeSelect={handleTreeSelect}
              defaultCollapseIcon={<ExpandMoreIcon />}
              defaultExpandIcon={<ChevronRightIcon />}
              className={classes.treeView}
            >
              {treeCategories.map((category) =>
                renderTreeCategory(category, categoryNameMap, 0, classes)
              )}
            </TreeView>
          ) : (
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
          )}
        </FormControl>
      </>
    )
  }

  /**
   * MultipleSelect render function
   */
  public render(): React.ReactNode {
    const { categories } = this.props
    const { treeCategories } = this.props
    const { classes } = this.props
    const { headerText } = this.props
    if (categories === null) {
      return null
    } else {
      return this.renderCategory(
        categories,
        treeCategories,
        classes,
        headerText
      )
    }
  }
}

export const ToolbarCategory = withStyles(categoryStyle, { withTheme: true })(
  MultipleSelect
)
export default ToolbarCategory
