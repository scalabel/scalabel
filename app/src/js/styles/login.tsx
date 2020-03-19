import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'

export const loginStyle = (theme: Theme) => createStyles({
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
    fontSize: '110%'
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1)
  },
  submit: {
    margin: theme.spacing(3, 0, 2)
  },
  bdd_font: {
    fontFamily: 'Exo+2',
    fontStyle: 'italic',
    fontWeight: 'normal'
  },
  scalabel_font: {
    fontWeight: 'normal'
  },
  btn: {
    color: 'lightgrey',
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
    transform: 'translate(-50%, -50%)'
  }
})
