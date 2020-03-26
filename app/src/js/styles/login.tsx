import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'

export const loginStyle = (theme: Theme) => createStyles({
  body: {
    backgroundColor: '#3B7EA1',
    width: '100%',
    height: '100%',
    position: 'fixed'
  },
  img: {
    margin: 0,
    position: 'absolute',
    top: '30%',
    left: '50%',
    transform: 'translate(-50%, -50%)'
  },
  break: {
    lineHeight: '200%'
  },
  link: {
    color: 'white',
    fontSize: '110%',
    margin: '0 10px'
  },
  intro: {
    fontWeight: 'bold',
    fontSize: '130%',
    textAlign: 'center',
    color: '#dddddd',
    whiteSpace: 'nowrap',

    margin: 0,
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)'
  },
  form: {
    width: '400px',
    marginTop: theme.spacing(1),
    background: 'white',
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    padding: '20px',
    borderRadius: '10px'
  },
  checkbox: {
    color: 'rgba(0, 0, 0, 0.54)'
  },
  bdd_font: {
    fontStyle: 'italic',
    fontWeight: 'normal',
    margin: '0 2px'
  },
  btn: {
    padding: '16px 32px',
    textAlign: 'center',
    textDecoration: 'none',
    display: 'inline-block',
    fontSize: '16px',
    cursor: 'pointer',
    margin: 0,
    position: 'absolute',
    top: '60%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    color: '#f5f5f5',
    backgroundColor: 'transparent' }
})

export const loginTextFieldStyle = () => createStyles({
  root: {
    'minHeight': '4.5rem',
    '& label.Mui-focused': {
      color: '#3B7EA1'
    },
    '& .MuiInput-underline:after': {
      borderBottomColor: '#3B7EA1'
    },
    '& .MuiOutlinedInput-root': {
      '& fieldset': {
        borderColor: 'red'
      },
      '&:hover fieldset': {
        borderColor: 'yellow'
      },
      '&.Mui-focused fieldset': {
        borderColor: '#3B7EA1'
      }
    }
  }
})

export const loginButtonStyle = (theme: Theme) => createStyles({
  root: {
    'backgroundColor': '#3B7EA1',
    'color': 'white',
    'margin': theme.spacing(3, 0, 2),
    '&:hover': {
      backgroundColor: '#418eb5',
      borderColor: '#0062cc',
      boxShadow: 'none'
    },
    '&:active': {
      boxShadow: 'none',
      backgroundColor: '#58b1e0',
      borderColor: '#005cbf'
    },
    '&:focus': {
      boxShadow: '0 0 0 0.2rem rgba(0,123,255,.5)'
    }
  }
})

export const loginCheckboxStyle = () => createStyles({
  root: {
    '&:hover': {
      backgroundColor: 'transparent',
      color: '#3B7EA1'
    }
  }
})
