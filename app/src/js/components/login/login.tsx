import Button from '@material-ui/core/Button'
import Checkbox from '@material-ui/core/Checkbox'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Grid from '@material-ui/core/Grid'
import Link from '@material-ui/core/Link'
import Modal from '@material-ui/core/Modal'
import { withStyles } from '@material-ui/core/styles'
import TextField from '@material-ui/core/TextField'
import React from 'react'
import { loginStyle } from '../../styles/login'

export interface ClassType {
  /** img */
  img: string,
  /** break */
  break: string,
  /** link */
  link: string,
  /** form */
  form: string,
  /** submit */
  submit: string,
  /** bdd_font */
  bdd_font: string,
  /** scalabel_font */
  scalabel_font: string,
  /** btn */
  btn: string
}

export interface Props {
  /** Create classes */
  classes: ClassType
}

export interface State {
  /** boolean to force reload of the sidebar project list */
  open: boolean
}

/**
 * Component which display the create page
 * @param {object} props
 * @return component
 */
class Login extends React.Component<Props, State> {

  public constructor (props: Props) {
    super(props)
    this.state = {
      open: false
    }
  }

  /**
   * renders the create page
   * @return component
   */
  public render () {
    return (
      <body>
        <img
          src='img/scalabel_white.svg'
          alt='Scalabel Logo'
          height='200' />
        <br />
        <p> A scalable open-sourced annotation web tool brought by
            <a href='https://deepdrive.berkeley.edu/'> Berkeley DeepDrive </a>
            <br />
            For more information: <a href='http://scalabel.ai'> Scalabel.ai</a>.
        </p>
        <br />
        <div>
          <button type='button' onClick={this.handleOpen}>
          Login
          </button>
        </div>
        <Modal
          aria-labelledby='simple-modal-title'
          aria-describedby='simple-modal-description'
          open={this.state.open}
          onClose={this.handleClose}
        >
          <form noValidate>
            <TextField
              variant='outlined'
              margin='normal'
              required
              fullWidth
              id='email'
              label='Email Address'
              name='email'
              autoComplete='email'
              autoFocus
            />
            <TextField
              variant='outlined'
              margin='normal'
              required
              fullWidth
              name='password'
              label='Password'
              type='password'
              id='password'
              autoComplete='current-password'
            />
            <FormControlLabel
              control={<Checkbox value='remember' color='primary' />}
              label='Remember me'
            />
            <Button
              type='submit'
              fullWidth
              variant='contained'
              color='primary'
            >
              Sign In
            </Button>
            <Grid container>
              <Grid item xs>
                <Link href='#' variant='body2'>
                  Forgot password?
                </Link>
              </Grid>
              <Grid item>
                <Link href='#' variant='body2'>
                  {"Don't have an account? Sign Up"}
                </Link>
              </Grid>
            </Grid>
          </form>
        </Modal>
      </body>
    )
  }

  /**
   * set modal login open
   * list
   */
  private handleOpen = () => {
    this.setState({ open : true })
  }

  /**
   * set modal login close
   */
  private handleClose = () => {
    this.setState({ open : false })
  }

}

/** export Create page */
export default withStyles(loginStyle, { withTheme: true })(Login)
