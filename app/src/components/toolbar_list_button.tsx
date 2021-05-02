import { ListItemText } from "@material-ui/core"
import List from "@material-ui/core/List"
import ListItem from "@material-ui/core/ListItem"
import { withStyles } from "@material-ui/core/styles"
import ToggleButton from "@material-ui/lab/ToggleButton"
import ToggleButtonGroup from "@material-ui/lab/ToggleButtonGroup"
import React from "react"

import { listButtonStyle, toggleButtonStyle } from "../styles/label"

interface ClassType {
  /** root class */
  root: string
  /** element containg toggle class */
  toggleContainer: string
  /** button group class */
  buttonGroup: string
  /** primary text class */
  primary: string
  /** toggle button class */
  toggleButton: string
}

interface Props {
  /** handles attribute toggling */
  handleAttributeToggle: (toggleName: string, alignment: string) => void
  /** gets alignment index for selected attribute */
  getAlignmentIndex: (toggleName: string) => number
  /** styles of ToggleButtons */
  classes: ClassType
  /** name of ToggleButtons */
  name: string
  /** values of ToggleButtons */
  values: string[]
}

/**
 * This is ToggleButtons component that displays
 * the everything post in the dashboard.
 *
 * @param {object} props
 */
class ToggleButtons extends React.Component<Props> {
  /**
   * handleAlignment of ToggleButtons that align buttons
   *
   * @param _event
   * @param alignment
   */
  public handleAlignment = (
    _event: React.MouseEvent<HTMLElement>,
    alignment: string
  ): void => {
    this.props.handleAttributeToggle(this.props.name, alignment)
    // Re-render to get correct alignment
    this.setState({})
  }

  /** render function of ToggleButtons */
  public render(): JSX.Element {
    const { name, classes, values } = this.props
    const ToggleBtn = withStyles(toggleButtonStyle)(ToggleButton)
    return (
      <List style={{ width: "100%", padding: "0px" }}>
        <ListItem dense={true} className={classes.primary}>
          <ListItemText className={classes.primary}>
            <div className={classes.primary}>{name}</div>
          </ListItemText>
        </ListItem>
        <ListItem dense={true} className={classes.toggleContainer}>
          <ToggleButtonGroup
            className={classes.buttonGroup}
            value={values[this.props.getAlignmentIndex(name)]}
            exclusive
            onChange={this.handleAlignment}
          >
            {values.map((element: string) => (
              <ToggleBtn
                className={classes.toggleButton}
                value={element}
                key={element}
                data-testid={"toggle-button-" + element}
              >
                {" "}
                {element}{" "}
              </ToggleBtn>
            ))}
          </ToggleButtonGroup>
        </ListItem>
      </List>
    )
  }
}

export const ListButton = withStyles(listButtonStyle)(ToggleButtons)
