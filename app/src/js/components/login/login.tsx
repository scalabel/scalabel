import Button from '@material-ui/core/Button'
import Checkbox from '@material-ui/core/Checkbox'
import CssBaseline from '@material-ui/core/CssBaseline'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import Grid from '@material-ui/core/Grid'
import Link from '@material-ui/core/Link'
import Modal from '@material-ui/core/Modal'
import { withStyles } from '@material-ui/core/styles'
import TextField from '@material-ui/core/TextField'
import React from 'react'
import { loginButtonStyle, loginCheckboxStyle, loginStyle, loginTextFieldStyle } from '../../styles/login'

export interface ClassType {
  /** body */
  body: string,
  /** img */
  img: string,
  /** break */
  break: string,
  /** link */
  link: string,
  /** intro */
  intro: string,
  /** form */
  form: string,
  /** checkbox */
  checkbox: string,
  /** bdd_font */
  bdd_font: string,
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
      <div className={this.props.classes.body}>
        <CssBaseline />
        <img
          src='img/scalabel_white.svg'
          className={this.props.classes.img}
          alt='Scalabel Logo'
          height='200' />
        <br className={this.props.classes.break}/>
        <p className={this.props.classes.intro}>
            A scalable open-sourced annotation web tool brought by
            <Link className=
              {this.props.classes.link + ' ' + this.props.classes.bdd_font}
              href='https://deepdrive.berkeley.edu/'
              variant='h6'>
                Berkeley DeepDrive
            </Link>
            <br />
            For more information:
            <Link className={this.props.classes.link} href='http://scalabel.ai'>
              Scalabel.ai
            </Link>
        </p>
        <br className={this.props.classes.break}/>
        <div>
          <Button className={this.props.classes.btn} onClick={this.handleOpen}>
          Login
          </Button>
        </div>
        <Modal
          open={this.state.open}
          onClose={this.handleClose}
        >
          <form className={this.props.classes.form} noValidate>
            <LoginText
              margin='normal'
              required
              fullWidth
              id='email'
              label='Email Address'
              name='email'
              autoComplete='email'
              autoFocus
            />
            <LoginText
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
              control={<LoginCheckbox value='remember' color='secondary' />}
              label='Remember me'
              className={this.props.classes.checkbox}
            />
            <LoginButton
              type='submit'
              fullWidth
            >
              Sign In
            </LoginButton>
            <Grid container>
              <Grid item xs>
                <Link href='/forget_password'>
                  Forgot password?
                </Link>
              </Grid>
            </Grid>
          </form>
        </Modal>
      </div>
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

const LoginText = withStyles(loginTextFieldStyle)(TextField)
const LoginButton = withStyles(loginButtonStyle)(Button)
const LoginCheckbox = withStyles(loginCheckboxStyle)(Checkbox)

/** export Create page */
export default withStyles(loginStyle, { withTheme: true })(Login)
