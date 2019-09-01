import React from 'react'
import { StyledButton } from '../styles/label'

/**
 * This is genButton function that renders the buttons in the toolbar.
 * @param {string} name name of the button
 * @param {function} clickCallback call back function for lick
 */
export function makeButton (name: string,
                            clickCallback: () => void = () => { return }) {
  return (
    <StyledButton onClick={clickCallback}>{name}</StyledButton>
  )
}
