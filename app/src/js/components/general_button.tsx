import React from 'react';
import {StyledButton} from '../styles/label';

/**
 * This is genButton function that renders the buttons in the toolbar.
 * @param props
 */
export function genButton(props: {
    /** name of the button */
    name: string
    }) {
    const {name} = props;
    return(
        <StyledButton>{name}</StyledButton>
    );
}
