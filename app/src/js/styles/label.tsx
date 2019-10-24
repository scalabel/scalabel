import { withStyles } from '@material-ui/core'
import Box from '@material-ui/core/Box/Box'
import Button from '@material-ui/core/Button/Button'
import { blue } from '@material-ui/core/colors'
import grey from '@material-ui/core/colors/grey'
import createStyles from '@material-ui/core/styles/createStyles'

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

export const StyledButton = withStyles({
  root: {
    borderRadius: 0,
    border: 0,
    color: 'black',
    height: '80%',
    width: '80%',
    padding: '5px 15px',
    boxShadow: '0 1px 0px 5px rgba(250, 250, 250, 1)',
    fontSize: '15px',
    background: 'white',
    margin: '0px 20px'
  },
  label: {
    fontSize: '15px'
  }
})(Button)

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
    height: '50px',
    left: '10px',
    position: 'relative',
    width: 'calc(100% - 40px)',
    top: 'calc(100% - 50px)'
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
