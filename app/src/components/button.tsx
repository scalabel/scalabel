import { Button, ButtonProps, WithStyles, withStyles } from "@material-ui/core"
import React from "react"
import { StyledButtonProps, styledButtonStyle } from "../styles/label"

/**
 * raw styled button.
 *
 * @param props props.
 */
function StyledButtonRaw(
  props: WithStyles<typeof styledButtonStyle> &
    Omit<ButtonProps, keyof StyledButtonProps> &
    StyledButtonProps
): JSX.Element {
  const { classes, ...other } = props
  return <Button className={classes.root} {...other} />
}

const StyledButton = withStyles(styledButtonStyle)(StyledButtonRaw)

/**
 * This is genButton function that renders the buttons in the toolbar.
 *
 * @param {string} name name of the button
 * @param {function} clickCallback call back function for lick
 * @param bgColor
 */
export function makeButton(
  name: string,
  clickCallback: () => void = () => {},
  bgColor: string = "white"
): JSX.Element {
  return (
    <StyledButton onClick={clickCallback} background={bgColor}>
      {name}
    </StyledButton>
  )
}
