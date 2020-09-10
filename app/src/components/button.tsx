import { Button, withStyles } from "@material-ui/core"
import React from "react"

import { toolbarButtonStyle } from "../styles/label"

interface ButtonClassType {
  /** flex grow buffer style */
  root: string
  /** class type for chip */
  label: string
}

interface ButtonProps {
  classes: ButtonClassType
  children: React.ReactNode
  onClick: () => void
}
/**
 * raw styled button.
 *
 * @param props props.
 */
function StyledButtonRaw(props: ButtonProps): JSX.Element {
  const { classes, children, onClick } = props
  return (
    <Button
      size="small"
      color="primary"
      variant="contained"
      className={classes.root}
      onClick={onClick}
    >
      {children}
    </Button>
  )
}

const StyledButton = withStyles(toolbarButtonStyle)(StyledButtonRaw)

/**
 * This is genButton function that renders the buttons in the toolbar.
 *
 * @param {string} name name of the button
 * @param {function} clickCallback call back function for lick
 * @param bgColor
 */
export function makeButton(
  name: string,
  clickCallback: () => void = () => {}
): JSX.Element {
  return <StyledButton onClick={clickCallback}>{name}</StyledButton>
}
