import React from 'react';
import {StyledButton} from '../styles/label';

export function genButton(props: {name: any}) {
    const {name} = props;
    return(
        <StyledButton>{name}</StyledButton>
    );
}
