import Box from '@material-ui/core/Box/Box'
import Button, { ButtonProps } from '@material-ui/core/Button'
import { blue, grey } from '@material-ui/core/colors'
import { createStyles, withStyles, WithStyles } from '@material-ui/core/styles'
import { Omit } from '@material-ui/types'
import React from 'react'

export const categoryStyle = () => createStyles({
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    flexDirection: 'column',
    alignItems: 'left'
  },
  formControl: {
    margin: '0px 4px',
    minWidth: 150,
    maxWidth: 360
  },
  primary: {
    fontSize: '15px'
  },
  checkbox: {
    'color': grey[600],
    '&$checked': {
      color: blue[500]
    },
    'fontSize': '15px',
    'margin': '-10px',
    'marginRight': '-5px'
  },
  checked: {}
})

export const switchStyle = () => ({
  root: {
    width: '100%',
    maxWidth: 360
  },
  primary: {
    fontSize: '15px'
  },
  switchBase: {
    'color': grey[400],
    '&$checked': {
      'color': grey[500],
      '& + $track': {
        backgroundColor: blue[700]
      }
    }
  },
  checked: {},
  track: {}
})

export interface StyledButtonProps {
  /**
   * background color. 
   * TODO: find a strict color type
   */
  background: string
}

const styledButtonStyle = createStyles({
  root: {
    borderRadius: 0,
    border: 0,
    color: 'black',
    height: '80%',
    width: '80%',
    padding: '5px 15px',
    boxShadow: '0 1px 0px 5px rgba(250, 250, 250, 1)',
    fontSize: '15px',
    background: (props: StyledButtonProps) => props.background,
    margin: '0px 20px'
  },
  label: {
    fontSize: '15px'
  }
})

/**
 * raw styled button
 * @param props props
 */
function StyledButtonRaw (
  props: WithStyles<typeof styledButtonStyle>
        & Omit<ButtonProps, keyof StyledButtonProps> & StyledButtonProps
) {
  const { classes, background, ...other } = props
  return <Button className={classes.root} {...other} />
}

export const StyledButton = withStyles(styledButtonStyle)(StyledButtonRaw)

export const toggleButtonStyle = () => ({
  root: {
    color: 'rgba(0, 0, 0, 0.38)',
    height: '28px',
    padding: '1px 2px',
    fontSize: '15px',
    minWidth: '28px',
    borderRadius: '2px'
  },
  label: {
    fontSize: '11px'
  }
})

export const listButtonStyle = () => ({
  root: {
    padding: '0px',
    height: '28px'
  },
  toggleContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    background: 'rgba(250,250,250,0)'
  },
  buttonGroup: {
    width: '100%',
    display: 'flex',
    // needed to prevent compiler from complaining about types
    flexWrap: 'wrap' as ('wrap')
  },
  primary: {
    fontSize: '15px'
  },
  toggleButton: {
    flexGrow: 1
  }
})

export const label2dViewStyle = () => createStyles({
  label2d_canvas: {
    'position': 'absolute',
    'z-index': 1
  },
  control_canvas: {
    'position': 'absolute',
    'visibility': 'hidden',
    'z-index': 2
  },
  hair: {
    'position': 'fixed',
    'margin-top': '0px',
    'margin-left': '0px',
    'background': 'transparent',
    'border-top': '1px dotted #0000ff',
    'border-left': '1px dotted #0000ff',
    'pointer-events': 'none',
    'z-index': 3
  }
})

export const imageViewStyle = () => createStyles({
  image_canvas: {
    'position': 'absolute',
    'z-index': 0
  }
})

export const playerControlStyles = () => createStyles({
  button: {
    color: '#bbbbbb',
    left: '-3px',
    verticalAlign: 'middle'
  },
  playerControl: {
    display: 'block',
    position: 'relative',
    top: 'calc(100% - 55px)',
    zIndex: 100
  },
  input: {
    background: '#000000',
    color: 'green',
    direction: 'rtl',
    width: '50px',
    fontWeight: 500,
    left: '-1px',
    right: '2px',
    verticalAlign: 'middle'
  },
  underline: {
    color: 'green'
  },
  slider: {
    selectionColor: 'green',
    rippleColor: 'white',
    verticalAlign: 'middle'
  }
})

export const LayoutStyles = () => createStyles({
  titleBar: {
    height: '50px'
  },
  main: {
    height: 'calc(100% - 50px)',
    display: 'block',
    position: 'absolute',
    outline: 'none',
    width: '100%'
  },
  interfaceContainer: {
    display: 'block',
    height: '100%',
    position: 'absolute',
    outline: 'none',
    width: '100%',
    background: '#222222'
  },
  paneContainer: {
    width: '100%',
    height: 'calc(100% - 60px)',
    position: 'absolute',
    top: 0,
    left: 0
  }
})

export const StatusMessageBox = withStyles({
  root: {
    padding: '3px',
    fontSize: 'fontSize',
    background: '#909090',
    marginLeft: '40px',
    borderRadius: '2px',
    color: 'black'
  }
})(Box)
