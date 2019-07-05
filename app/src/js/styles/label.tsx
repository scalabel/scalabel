import { Theme, withStyles } from '@material-ui/core'
import Button from '@material-ui/core/Button/Button'
import { blue } from '@material-ui/core/colors'
import grey from '@material-ui/core/colors/grey'
import createStyles from '@material-ui/core/styles/createStyles'

export const categoryStyle = (theme: Theme) => createStyles({
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'left'
  },
  formControl: {
    margin: theme.spacing(1),
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
    'margin': '-10px'
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
  colorSwitchBase: {
    'color': grey[400],
    '&$colorChecked': {
      'color': grey[500],
      '& + $colorBar': {
        backgroundColor: grey[600]
      }
    }
  },
  colorBar: {},
  colorChecked: {}
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
    textTransform: 'uppercase',
    fontSize: '15px'
  },
  itemText: {
    fontSize: 10,
    fontWeight: 500
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
    height: '28px'
  },
  toggleContainer: {
    height: '28px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    background: 'rgba(250,250,250,0)'
  },
  buttonGroup: {
    height: '28px',
    width: '100%'
  },
  primary: {
    fontSize: '15px'
  }
})

export const imageViewStyle = () => createStyles({
  canvas: {
    position: 'absolute'
  },
  mask: {
    display: 'block',
    height: 'calc(100% - 20px)',
    top: '10px', left: '10px',
    position: 'absolute', overflow: 'scroll',
    outline: 'none',
    width: 'calc(100% - 20px)'
  },
  background: {
    display: 'block', height: 'calc(100% - 50px)',
    position: 'absolute',
    outline: 'none', width: '100%', background: '#222222'
  }
})
